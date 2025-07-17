# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 11:45:31 2025

@author: ariad
"""

# classify_one_workbook.py
import math, time, pandas as pd, torch
from sentence_transformers import SentenceTransformer

# ─── CONFIG ──────────────────────────────────────────────────────────────
XLSX        = r"C:/Users/ariad/Desktop/COT_penjar/Nova Proposta/protesis_cot.xlsx"
SHEET_REG   = "Regist_m1"        # full amb: Codi_HSPAU, Descripcio_HSPAU
SHEET_CLS   = "Classif"   # full amb: Codi_Classif, ICS_Grup_article, Descripcio_Classif
#MODEL_DIR   = r"C:/Users/ariad/Desktop/Millora de l'entrenament/model_turbo_m144"
MODEL_DIR   = r"C:/Users/ariad/Desktop/COT_penjar/Nova Proposta/fine_tuned_model_150_epoca3"
base_model_name="sentence-transformers/distiluse-base-multilingual-cased-v1"
OUT_SHEET   = "resultat_01"      # full nou (se sobreescriurà si ja existeix)

BLOCK_SIZE  = 4_000           # files de Regist per bloc (ajusta segons RAM/VRAM)
USE_FP16    = True            # posa False si vas amb CPU
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    # 1) Llegeix les dues fulles del mateix Excel
    df_cls = pd.read_excel(
        XLSX, sheet_name=SHEET_CLS,
        names=["Codi_Classif", "ICS_Grup_article", "Descripcio_Classif"]
    )
    df_reg = pd.read_excel(XLSX, sheet_name=SHEET_REG)   # Codi_HSPAU, Descripcio_HSPAU

    # 2) Carrega el model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SentenceTransformer(MODEL_DIR, device=device)

    # 3) Embeddings de Classif
    dtype = torch.float16 if (device == "cuda" and USE_FP16) else torch.float32
    cls_emb = model.encode(
        df_cls["Descripcio_Classif"].astype(str).tolist(),
        batch_size=512, convert_to_tensor=True,
        normalize_embeddings=True, dtype=dtype
    )                                                    # [N_cls, d]

    # 4) Processem Regist per blocs
    results, n_blocks = [], math.ceil(len(df_reg) / BLOCK_SIZE)
    print(f"→ Processem {len(df_reg)} registres en {n_blocks} blocs…")

    for b in range(n_blocks):
        s, e = b * BLOCK_SIZE, min((b + 1) * BLOCK_SIZE, len(df_reg))
        reg_texts = df_reg["Descripcio_HSPAU"].astype(str).iloc[s:e].tolist()

        reg_emb = model.encode(
            reg_texts, batch_size=512, convert_to_tensor=True,
            normalize_embeddings=True, dtype=dtype
        )                                                # [block, d]

        sims          = reg_emb @ cls_emb.T              # producte escalar = cos sim
        sim_vals, idx = torch.max(sims, dim=1)

        block_df = pd.DataFrame({
            "Codi_HSPAU"            : df_reg["Codi_HSPAU"].iloc[s:e].values,
            "Descripcio_HSPAU"      : reg_texts,
            "Codi_Classif_Pred"     : df_cls["Codi_Classif"].iloc[idx.cpu()].values,
            "ICS_Grup_article_Pred" : df_cls["ICS_Grup_article"].iloc[idx.cpu()].values,
            "Descripcio_Classif_Pred": df_cls["Descripcio_Classif"].iloc[idx.cpu()].values,
            "Similitud"             : sim_vals.cpu().numpy()
        })
        results.append(block_df)
        print(f"  bloc {b+1}/{n_blocks} ✔  ({s}–{e-1})  {time.time()-t0:,.1f}s")

    out_df = pd.concat(results, ignore_index=True)

    # 5) Escriu a un nou full del mateix llibre
    with pd.ExcelWriter(XLSX, engine="openpyxl", mode="a", if_sheet_exists="replace") as wr:
        out_df.to_excel(wr, sheet_name=OUT_SHEET, index=False)

    print(f"\n✓ Full '{OUT_SHEET}' creat/actualitzat – {time.time()-t0:,.1f}s totals")

if __name__ == "__main__":
    main()
