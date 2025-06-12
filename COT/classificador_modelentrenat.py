# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 23:37:29 2025

@author: ariad
"""

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

def classify_regist_entire(
    classif_cleaned_path=r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/Classif_cleaned.xlsx",
    regist_cleaned_path=r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/Regist_cleaned.xlsx",
    model_path=r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/fine_tuned_model_150_epoca4",
    output_excel_path=r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/results_150_epoca4.xlsx"
):
    """
    1) Carrega tot 'Regist_cleaned.xlsx' i 'Classif_cleaned.xlsx'.
    2) Per cada fila de Regist, troba en Classif la millor coincidència (similitud cosinus).
    3) Desa un Excel 'regist_classification_results.xlsx' amb:
         - Codi_HSPAU, Descripcio_HSPAU
         - Codi_Classif_Pred, ICS_Grup_article_Pred, Descripcio_Classif_Pred
         - Similitud
    4) Utilitza el model entrenat indicat a 'model_path'.
    """

    # 1) Carregar dades
    print("Carregant fitxers d'entrada...")
    df_classif = pd.read_excel(classif_cleaned_path)
    df_regist = pd.read_excel(regist_cleaned_path)

    # Ajusta si cal: el fitxer Classif_cleaned.xlsx conté:
    #   [Codi_Classif, Grup_Articles, Descripcio_Classif]
    # o potser: [Codi_ICS, ICS_Grup_article, Descripcio_Classif]
    # Reanomenem segons convingui (exemple):
    df_classif.columns = ["Codi_ICS", "ICS_Grup_article", "Descripcio_Classif"]

    # Ajusta si cal: el fitxer Regist_cleaned.xlsx conté:
    #   [Codi_HSPAU, Descripcio_HSPAU]
    # Si hi ha més columnes, ignora-les o fes el que necessitis.

    # 2) Carregar el model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregant el model des de: {model_path} (dispositiu={device})")
    model = SentenceTransformer(model_path, device=device)

    # 3) Generar embeddings de TOT Classif un sol cop
    print("Generant embeddings per a totes les descripcions de Classif...")
    classif_texts = df_classif["Descripcio_Classif"].astype(str).tolist()
    classif_embeddings = model.encode(classif_texts, convert_to_tensor=True)

    # Llistes per a resultats
    codi_ics_pred_list = []
    ics_grup_article_pred_list = []
    desc_classif_pred_list = []
    sim_list = []

    # 4) Classificar cada fila de Regist
    print("Classificant tot Regist_cleaned...")
    for idx, row in df_regist.iterrows():
        codi_hpau = row["Codi_HSPAU"]
        desc_hpau = str(row["Descripcio_HSPAU"])

        # Embedding de la descripció del Regist
        emb_reg = model.encode(desc_hpau, convert_to_tensor=True)

        # Similitud cosinus amb tots els embeddings de Classif
        sims = util.cos_sim(emb_reg, classif_embeddings)[0]
        max_sim_val, max_idx = torch.max(sims, dim=0)
        max_idx = int(max_idx.item())

        # Recuperar fila Classif amb millor similitud
        best_row = df_classif.iloc[max_idx]
        codi_ics_pred = best_row["Codi_ICS"]
        grup_art_pred = best_row["ICS_Grup_article"]
        desc_classif_pred = best_row["Descripcio_Classif"]
        sim_score = float(max_sim_val.item())

        codi_ics_pred_list.append(codi_ics_pred)
        ics_grup_article_pred_list.append(grup_art_pred)
        desc_classif_pred_list.append(desc_classif_pred)
        sim_list.append(sim_score)

    # 5) Afegir columnes noves al df_regist
    df_regist["Codi_ICS_Pred"] = codi_ics_pred_list
    df_regist["ICS_Grup_article_Pred"] = ics_grup_article_pred_list
    df_regist["Descripcio_Classif_Pred"] = desc_classif_pred_list
    df_regist["Similitud"] = sim_list

    # 6) Desa un Excel amb resultats
    df_regist.to_excel(output_excel_path, index=False)
    print(f"\nResultats desats a: {output_excel_path}")
    print(f"Total files de Regist processades: {len(df_regist)}.")

def main():
    classif_cleaned_path = r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/Classif_cleaned.xlsx"
    regist_cleaned_path  = r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/Regist_cleaned.xlsx"
    model_path           = r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/fine_tuned_model_150_epoca4"
    output_excel_path    = r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/results_150_epoca4.xlsx"

    classify_regist_entire(
        classif_cleaned_path=classif_cleaned_path,
        regist_cleaned_path=regist_cleaned_path,
        model_path=model_path,
        output_excel_path=output_excel_path
    )

if __name__ == "__main__":
    main()
