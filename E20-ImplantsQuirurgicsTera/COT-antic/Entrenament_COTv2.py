# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 18:55:31 2025

@author: ariad
"""

# -*- coding: utf-8 -*-
"""
Adaptat per a ús multiplataforma i Colab. Fa fine-tuning amb checkpoints.
"""
import os
import re
import time
import random
from pathlib import Path
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

# ────────────────────────────────
# CONFIGURACIÓ DE DIRECTORIS
# ────────────────────────────────
IS_COLAB = 'google.colab' in str(get_ipython())
BASE_DIR = Path().resolve() if IS_COLAB else Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

VALIDATED_FILE = DATA_DIR / "Mostra_cleaned.xlsx"
CLASSIF_FILE   = DATA_DIR / "Classif_cleaned.xlsx"

MODEL_FOLDER         = DATA_DIR / "fine_tuned_model_150_epoca3"
OUTPUT_MODEL         = DATA_DIR / "fine_tuned_model_150_epoca5"
CHECKPOINT_DIR       = DATA_DIR / "checkpoints_150_epoca5"

# ────────────────────────────────
def load_and_prepare_data_all_descripcions(validated_path, classif_cleaned_path):
    df_val = pd.read_excel(validated_path)
    df_classif = pd.read_excel(classif_cleaned_path)
    df_classif.columns = ["Codi_Classif", "Grup_Articles", "Descripcio_Classif"]

    training_examples = []
    for _, row_val in df_val.iterrows():
        desc_hpau = str(row_val["Descripcio_HSPAU"])
        codi_classif_val = str(row_val["Codi_Classif"])

        sub_classif = df_classif[df_classif["Codi_Classif"] == codi_classif_val]
        for _, row_cls in sub_classif.iterrows():
            desc_classif = str(row_cls["Descripcio_Classif"])
            training_examples.append(InputExample(texts=[desc_hpau, desc_classif]))
    return training_examples

def find_last_checkpoint(checkpoint_dir):
    if not checkpoint_dir.exists():
        return (1, 0)
    pattern = re.compile(r"checkpoint_epoch(\d+)_chunk(\d+)$")
    max_epoch, max_chunk = 1, 0
    for name in os.listdir(checkpoint_dir):
        match = pattern.match(name)
        if match:
            ep, ch = int(match.group(1)), int(match.group(2))
            if ep > max_epoch or (ep == max_epoch and ch > max_chunk):
                max_epoch, max_chunk = ep, ch
    return (max_epoch, max_chunk)

def compute_simple_metric(model, examples, n=500):
    if len(examples) == 0:
        return 0.0
    sample = random.sample(examples, min(n, len(examples)))
    sims = []
    for ex in sample:
        embA = model.encode(ex.texts[0], convert_to_tensor=True)
        embB = model.encode(ex.texts[1], convert_to_tensor=True)
        sim = util.cos_sim(embA, embB)[0].item()
        sims.append(sim)
    return sum(sims) / len(sims)

def train_in_chunks(
    model_folder,
    training_examples,
    epochs=1,
    lr=2e-5,
    batch_size=8,
    chunk_size=10000,
    warmup_fixed=500,
    output_trained_model="fine_tuned_model",
    checkpoint_dir="checkpoints"
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    last_epoch, last_chunk = find_last_checkpoint(checkpoint_dir)
    print(f"Últim checkpoint trobat => epoch={last_epoch}, chunk={last_chunk}")

    if last_epoch > 1 or last_chunk > 0:
        chk_path = checkpoint_dir / f"checkpoint_epoch{last_epoch}_chunk{last_chunk}"
        if chk_path.exists():
            print(f"Carregant checkpoint existent: {chk_path}")
            model = SentenceTransformer(str(chk_path), device=device)
        else:
            print(f"No trobat {chk_path}; es carrega model base: {model_folder}")
            model = SentenceTransformer(str(model_folder), device=device)
    else:
        print(f"Carregant model base: {model_folder}")
        model = SentenceTransformer(str(model_folder), device=device)

    random.shuffle(training_examples)
    total_examples = len(training_examples)
    n_chunks_per_epoch = (total_examples // chunk_size) + (1 if total_examples % chunk_size != 0 else 0)
    epoch_start = last_epoch
    chunk_start = last_chunk + 1 if last_chunk > 0 else 1

    for e in range(epoch_start, epochs + 1):
        print(f"\n==== Època {e}/{epochs} ====")
        if e > epoch_start:
            random.shuffle(training_examples)

        current_index = (e == epoch_start) * (last_chunk * chunk_size)
        chunk_id = chunk_start if e == epoch_start else 1

        while current_index < total_examples:
            end_index = min(current_index + chunk_size, total_examples)
            chunk_data = training_examples[current_index:end_index]

            print(f"\n Chunk {chunk_id}: de {current_index} a {end_index} (total {len(chunk_data)})")
            num_chunks_left = n_chunks_per_epoch - (chunk_id - 1)
            start_time = time.time()

            train_dataloader = DataLoader(chunk_data, shuffle=True, batch_size=batch_size)
            train_loss = losses.MultipleNegativesRankingLoss(model)
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1,
                warmup_steps=warmup_fixed,
                optimizer_params={'lr': lr},
                show_progress_bar=True
            )

            elapsed = time.time() - start_time
            metric_val = compute_simple_metric(model, chunk_data, n=300)
            print(f"Similitud mitjana (subset): {metric_val:.4f}")

            chk_path = checkpoint_dir / f"checkpoint_epoch{e}_chunk{chunk_id}"
            model.save(str(chk_path))
            print(f"Checkpoint desat a: {chk_path}")
            est_time_left = elapsed * (num_chunks_left - 1)
            print(f"Temps chunk actual: {elapsed/60:.2f} min. ETA: ~{est_time_left/60:.2f} min.")

            current_index = end_index
            chunk_id += 1
        chunk_start = 1

    model.save(str(output_trained_model))
    print(f"\n✅ Model fine-tunat desat a: {output_trained_model}")
    return model

# ────────────────────────────────
def main():
    print("📦 Carregant dades i configuració...")

    training_examples = load_and_prepare_data_all_descripcions(
        validated_path=VALIDATED_FILE,
        classif_cleaned_path=CLASSIF_FILE
    )
    print(f"S'han creat {len(training_examples)} exemples positius.")

    model = train_in_chunks(
        model_folder=MODEL_FOLDER,
        training_examples=training_examples,
        epochs=1,
        lr=2e-5,
        batch_size=8,
        chunk_size=10000,
        warmup_fixed=500,
        output_trained_model=OUTPUT_MODEL,
        checkpoint_dir=CHECKPOINT_DIR
    )

if __name__ == "__main__":
    main()
