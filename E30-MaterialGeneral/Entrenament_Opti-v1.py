# -*- coding: utf-8 -*-
"""
Created on Mon May  5 20:50:08 2025

@author: ariad
"""

# turbo_finetune.py
import os
import time
import random
import pandas as pd          # ★ separa els imports; “import …, pandas as pd” donava error
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# ─── CONFIG ─────────────────────────────────────────────
XLSX          = r"C:/Users/ariad/Documents/GitHub/CatalegML/E30-MaterialGeneral/Cataleg_m144.xlsx"
SHEET_VAL     = "entreno"
SHEET_CLS     = "classif_E30"

BASE_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
OUT_DIR       = r"C:/Users/ariad/Documents/GitHub/CatalegML/E30-MaterialGeneral/model_turbo_m144"

FREEZE_N_LAYERS = 5     # congela les 5 primeres capes
BATCH_SIZE_GPU  = 512
BATCH_SIZE_CPU  = 64
LR              = 8e-4
EPOCHS          = 1
AMP             = True
# ───────────────────────────────────────────────────────


# ---------- 1. Carrega parelles ------------------------------------------------
def load_pairs(sheet_val, sheet_cls):
    df_val = pd.read_excel(XLSX, sheet_name=sheet_val)

    # Indica que la primera fila ja és capçalera i reanomena columnes
    df_cls = pd.read_excel(
        XLSX, sheet_name=sheet_cls,
        names=["Codi_Classif", "Grup", "Descr"], header=0
    )

    pairs = []
    for _, r in df_val.iterrows():
        subset = df_cls[df_cls.Codi_Classif == str(r.Codi_Classif)]
        for _, s in subset.iterrows():
            pairs.append(
                InputExample(
                    texts=[str(r.Descripcio_HSPAU), str(s.Descr)]
                )
            )
    return pairs


# ---------- 2. Construeix el model --------------------------------------------
def build_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(BASE_MODEL, device=device)

    # ➊ Congela capes del Transformer
    for name, param in model[0].auto_model.named_parameters():
        if name.startswith("encoder.layer"):
            layer_idx = int(name.split(".")[2])
            param.requires_grad = layer_idx >= FREEZE_N_LAYERS
        else:  # embeddings
            param.requires_grad = False

    # ➋ Descongela Pooling (canvi compatible amb v2.6+)  ★
    for mod_name, module in model._modules.items():
        if "pooling" in mod_name.lower():
            for p in module.parameters():
                p.requires_grad = True
            break  # trobada

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Paràmetres entrenables: {trainable/1e6:.1f} M")
    return model


# ---------- 3. Entrenament -----------------------------------------------------
def main():
    t0    = time.time()
    pairs = load_pairs(SHEET_VAL, SHEET_CLS)
    random.shuffle(pairs)
    print("Parelles carregades:", len(pairs))

    model  = build_model()
    device = model.device
    batch  = BATCH_SIZE_GPU if device.type == "cuda" else BATCH_SIZE_CPU

    dl   = DataLoader(
        pairs,
        shuffle=True,
        batch_size=batch,
        pin_memory=(device.type == "cuda")
    )
    loss = losses.MultipleNegativesRankingLoss(model)

    use_amp = AMP and device.type == "cuda"
    model.fit(
        train_objectives      = [(dl, loss)],
        epochs                = EPOCHS,
        optimizer_params      = {"lr": LR},
        use_amp               = use_amp,
        warmup_steps          = 0,
        show_progress_bar     = True
    )

    os.makedirs(os.path.dirname(OUT_DIR), exist_ok=True)   # ★ crea carpeta si cal
    model.save(OUT_DIR)
    print(f"√ Guardat a {OUT_DIR} – temps total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()