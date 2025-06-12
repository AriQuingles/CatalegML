# -*- coding: utf-8 -*-
"""
Created on Mon May  5 20:50:08 2025

@author: ariad
"""


import os
import re
import time
import random
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader

def load_and_prepare_data_all_descripcions(path, validated_path, classif_cleaned_path):
    """
    Carrega la mostra validada i Classif, i genera tants
    InputExamples com descripcions ICS hi hagi per un codi concret.
    """
    #df_cls = pd.read_excel(classif_xlsx, sheet_name=sheet_classif)
    df_val = pd.read_excel(path, sheet_name=validated_path)
    df_classif = pd.read_excel(path, sheet_name=classif_cleaned_path)
    df_classif.columns = ["Codi_Classif", "Grup_Articles", "Descripcio_Classif"]

    training_examples = []
    for _, row_val in df_val.iterrows():
        codi_hpau   = str(row_val["Codi_HSPAU"])
        desc_hpau   = str(row_val["Descripcio_HSPAU"])
        codi_classif_val = str(row_val["Codi_Classif"])

        sub_classif = df_classif[df_classif["Codi_Classif"] == codi_classif_val]
        if sub_classif.empty:
            continue

        for _, row_cls in sub_classif.iterrows():
            desc_classif = str(row_cls["Descripcio_Classif"])
            training_examples.append(InputExample(texts=[desc_hpau, desc_classif]))

    return training_examples


def find_last_checkpoint(checkpoint_dir):
    """
    Busca l’últim checkpoint en la forma checkpoint_epochX_chunkY
    Retorna (epoch, chunk) si el troba, si no (1, 0).
    """
    if not os.path.exists(checkpoint_dir):
        return (1, 0)

    pattern = re.compile(r"checkpoint_epoch(\d+)_chunk(\d+)$")
    max_epoch = 1
    max_chunk = 0
    for name in os.listdir(checkpoint_dir):
        match = pattern.match(name)
        if match:
            ep = int(match.group(1))
            ch = int(match.group(2))
            if ep > max_epoch or (ep == max_epoch and ch > max_chunk):
                max_epoch = ep
                max_chunk = ch
    return (max_epoch, max_chunk)


def compute_simple_metric(model, examples, n=500):
    """
    Petit test: agafa n exemples i calcula la similitud mitjana de la parella [textA, textB].
    """
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
    chunk_size=10000,              # Chunks més petits (10k)
    warmup_fixed=500,
    output_trained_model="fine_tuned_model",
    checkpoint_dir="checkpoints"
):
    """
    Entrena escalonadament, desant checkpoints i permetent reprendre.
    - chunk_size=10000 => processa 10k exemples per bloc -> més freqüent i segur
    - warmup_fixed=500 => warmup_steps fix
    - epochs=1 => 1 època
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1) Comprova si hi ha un checkpoint anterior
    last_epoch, last_chunk = find_last_checkpoint(checkpoint_dir)
    print(f"Últim checkpoint trobat => epoch={last_epoch}, chunk={last_chunk}")

    # 2) Carregar model
    if last_epoch > 1 or last_chunk > 0:
        # Hi ha un checkpoint posterior a l’epoch 1 o chunk >0
        chk_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{last_epoch}_chunk{last_chunk}")
        if os.path.exists(chk_path):
            print(f"Carregant checkpoint existent: {chk_path}")
            model = SentenceTransformer(chk_path, device=device)
        else:
            # Si no existeix el directori, torna al model base
            print(f"No trobat {chk_path}; es carrega model base: {model_folder}")
            model = SentenceTransformer(model_folder, device=device)
    else:
        print(f"Carregant model base: {model_folder}")
        model = SentenceTransformer(model_folder, device=device)

    # 3) Preparar dades
    random.shuffle(training_examples)
    total_examples = len(training_examples)
    print(f"Total d'exemples: {total_examples}")
    n_chunks_per_epoch = (total_examples // chunk_size) + (1 if total_examples % chunk_size != 0 else 0)
    print(f"~{n_chunks_per_epoch} chunks per epoch, chunk_size={chunk_size}")

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
            print(f"(Queden ~{num_chunks_left} chunks per acabar aquesta època)")

            start_time = time.time()
            train_dataloader = DataLoader(chunk_data, shuffle=True, batch_size=batch_size)
            train_loss = losses.MultipleNegativesRankingLoss(model)

            # warmup_steps fix -> no tan gran
            warmup_steps = warmup_fixed
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1,
                warmup_steps=warmup_steps,
                optimizer_params={'lr': lr},
                show_progress_bar=True
            )
            elapsed = time.time() - start_time

            # Mètrica de control
            metric_val = compute_simple_metric(model, chunk_data, n=300)
            print(f"Similitud mitjana (subset) = {metric_val:.4f}")

            # Desar checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{e}_chunk{chunk_id}")
            model.save(checkpoint_path)
            print(f"Checkpoint desat a: {checkpoint_path}")

            avg_time_per_chunk = elapsed
            est_time_left = avg_time_per_chunk * (num_chunks_left - 1)
            print(f"Temps chunk actual: {elapsed/60:.2f} min. ETA restants: ~{est_time_left/60:.2f} min.")

            current_index = end_index
            chunk_id += 1

        chunk_start = 1
        last_chunk = 0

    # 4) Desa el model final
    model.save(output_trained_model)
    print(f"\nModel fi-ne-tunat desat a: {output_trained_model}")

    return model


def main():
    path = r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/Cataleg_m144.xlsx"
    validated_path       = "entreno_200"
    classif_cleaned_path = "classif_E30"

    model_folder         = r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/fine_tuned_model_all_desc"
    output_trained_model = r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/fine_tuned_model_m144_e1"
    checkpoint_dir       = "checkpoints_m144_e1"

    # 1) Crear tots els exemples
    training_examples = load_and_prepare_data_all_descripcions(path, validated_path, classif_cleaned_path)
    print(f"S'han creat {len(training_examples)} exemples positius.")

    # 2) Entrenament escalonat
    model = train_in_chunks(
        model_folder=model_folder,
        training_examples=training_examples,
        epochs=1,           # Només 1 època
        lr=2e-5,
        batch_size=8,
        chunk_size=10000,   # Xunks més petits (10k)
        warmup_fixed=500,
        output_trained_model=output_trained_model,
        checkpoint_dir=checkpoint_dir
    )

if __name__ == "__main__":
    main()