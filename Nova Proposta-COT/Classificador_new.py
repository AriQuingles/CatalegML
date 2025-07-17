# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 11:38:39 2025

@author: ariad
"""
# eval_protesis_cot.py
import pandas as pd
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# ─── CONFIG ──────────────────────────────────────────────────────────────
XLSX         = Path(r"C:/Users/ariad/Desktop/COT_penjar/Nova Proposta/protesis_cot.xlsx")
SHEET_REG    = "RevisioVictor"     # Codi_HSPAU · Descripcio_HSPAU
SHEET_CLS    = "Classif"       # Codi_Classif · ICS_Grup_article · Descripcio_Classif
MODEL_DIR    = Path(r"C:/Users/ariad/Desktop/COT_penjar/Nova Proposta/fine_tuned_model_150_epoca3")
BASE_MODEL   = "sentence-transformers/distiluse-base-multilingual-cased-v1"
OUT_SHEET    = "resultat_RevisioVictor"   # es reemplaçarà si ja existeix
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    # 1) Carrega dades
    print("▶ Llegint pestanyes...")
    df_reg   = pd.read_excel(XLSX, sheet_name=SHEET_REG)
    df_cls   = pd.read_excel(XLSX, sheet_name=SHEET_CLS,
                             names=["Codi_Classif", "ICS_Grup_article", "Descripcio_Classif"])

    # 2) Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = SentenceTransformer(MODEL_DIR.as_posix(), device=device)
        print(f"✓ Model carregat de «{MODEL_DIR}»")
    except Exception:
        print("No s’ha trobat cap model fi-ne-tunat; es carrega el base d’Hugging Face…")
        model = SentenceTransformer(BASE_MODEL, device=device)

    # 3) Embeddings de Classif
    print("▶ Calculant embeddings de Classif…")
    cls_emb = model.encode(df_cls["Descripcio_Classif"].astype(str).tolist(),
                           convert_to_tensor=True)

    # 4) Classificació fila a fila
    cod_pred, grp_pred, desc_pred, sim_list = [], [], [], []

    print("▶ Classificant registres…")
    for desc in df_reg["Descripcio_HSPAU"].astype(str):
        emb = model.encode(desc, convert_to_tensor=True)
        sims = util.cos_sim(emb, cls_emb)[0]
        idx  = int(torch.argmax(sims))
        best = df_cls.iloc[idx]

        cod_pred.append(best["Codi_Classif"])
        grp_pred.append(best["ICS_Grup_article"])
        desc_pred.append(best["Descripcio_Classif"])
        sim_list.append(float(sims[idx]))

    # 5) Combina i desa
    df_out = df_reg.copy()
    df_out["Codi_Classif_Pred"]     = cod_pred
    df_out["ICS_Grup_article_Pred"] = grp_pred
    df_out["Descripcio_Classif_Pred"] = desc_pred
    df_out["Similitud"]             = sim_list

    mode = "a" if XLSX.exists() else "w"          # (normalment ja existeix)
    with pd.ExcelWriter(XLSX, engine="openpyxl", mode=mode,
                        if_sheet_exists="replace") as wr:
        df_out.to_excel(wr, sheet_name=OUT_SHEET, index=False)

    print(f"✓ Resultats escrits a «{XLSX.name}» (full «{OUT_SHEET}»)")
    print(f"   Files processades: {len(df_out)}")

if __name__ == "__main__":
    main()

