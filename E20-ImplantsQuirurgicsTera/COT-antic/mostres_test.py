# -*- coding: utf-8 -*-
"""
Adaptat per usar paths relatius i cross-platform amb pathlib.
Amb informació detallada de cada pas.
"""
from pathlib import Path
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

# ────────────────────────────────────────────────────────────────────────────────
# Configuració de rutes – relatives al directori del script
# ────────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

MOSTRA_FILE   = DATA_DIR / "Mostra_cleaned.xlsx"
CLASSIF_FILE  = DATA_DIR / "Classif_cleaned.xlsx"
MODEL_DIR     = DATA_DIR / "fine_tuned_model_150_epoca3"
OUTPUT_FILE   = DATA_DIR / "evaluation_results150_epocaA.xlsx"

# ────────────────────────────────────────────────────────────────────────────────
def evaluate_mostra_model(
    mostra_clean_path: Path = MOSTRA_FILE,
    classif_cleaned_path: Path = CLASSIF_FILE,
    model_path: Path = MODEL_DIR,
    output_excel_path: Path = OUTPUT_FILE,
    base_model_name: str = "sentence-transformers/distiluse-base-multilingual-cased-v1",
) -> None:
    """
    Avaluació d'un model fine-tuned sobre una mostra i classificació.
    """

    print("\n────── Ruta d'execució ──────")
    print(f"Script executat des de: {Path.cwd()}")
    print(f"Ruta absoluta de l'script (__file__): {BASE_DIR}")
    print(f"Fitxer mostra: {mostra_clean_path}")
    print(f"Fitxer classificació: {classif_cleaned_path}")
    print(f"Directori model: {model_path}")
    print(f"Fitxer de sortida: {output_excel_path}")
    print("──────────────────────────────\n")

    if not model_path.exists():
        raise FileNotFoundError(f"El directori del model no existeix: {model_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregant el model des de: {model_path}")
    model = SentenceTransformer(str(model_path), device=device)

    df_mostra = pd.read_excel(mostra_clean_path)
    df_classif = pd.read_excel(classif_cleaned_path)
    df_classif.columns = ["Codi_Classif", "Grup_Articles", "Descripcio_Classif"]

    classif_embeddings = model.encode(
        df_classif["Descripcio_Classif"].tolist(), convert_to_tensor=True
    )

    codi_classif_pred_list, grup_article_pred_list = [], []
    descripcio_pred_list, sim_list, encert_list = [], [], []

    for idx, row in df_mostra.iterrows():
        desc_hpau = str(row["Descripcio_HSPAU"])
        codi_classif_real = str(row["Codi_Classif"])

        emb_hpau = model.encode(desc_hpau, convert_to_tensor=True)
        sims = util.cos_sim(emb_hpau, classif_embeddings)[0]
        max_sim_val, max_idx = torch.max(sims, dim=0)
        best_row = df_classif.iloc[int(max_idx)]

        codi_classif_pred = best_row["Codi_Classif"]
        grup_article_pred = best_row["Grup_Articles"]
        desc_classif_pred = best_row["Descripcio_Classif"]
        sim_score = float(max_sim_val)

        codi_classif_pred_list.append(codi_classif_pred)
        grup_article_pred_list.append(grup_article_pred)
        descripcio_pred_list.append(desc_classif_pred)
        sim_list.append(sim_score)
        encert_list.append(codi_classif_pred == codi_classif_real)

        # 🟨 Informació detallada per cada fila
        print(f"[{idx+1}] Descripció: {desc_hpau}")
        print(f"    → Codi real: {codi_classif_real}")
        print(f"    → Predicció: {codi_classif_pred} ({grup_article_pred})")
        print(f"    → Descripció pred: {desc_classif_pred}")
        print(f"    → Similitud: {sim_score:.4f}")
        print(f"    → Encert: {'✅' if codi_classif_pred == codi_classif_real else '❌'}\n")

    df_mostra["Codi_Classif_Pred"] = codi_classif_pred_list
    df_mostra["ICS_Grup_article_Pred"] = grup_article_pred_list
    df_mostra["Descripcio_Classif_Pred"] = descripcio_pred_list
    df_mostra["Similitud"] = sim_list
    df_mostra["Encert"] = encert_list

    df_mostra.to_excel(output_excel_path, index=False)
    print(f"📁 Resultats desats a: {output_excel_path}")

    accuracy = sum(encert_list) / len(encert_list) * 100
    print(f"\n✅ Accuracy aproximada: {accuracy:.2f}%")
    print(f"🎯 {sum(encert_list)} encerts de {len(encert_list)} mostres totals.")

# ────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    evaluate_mostra_model()

if __name__ == "__main__":
    main()
