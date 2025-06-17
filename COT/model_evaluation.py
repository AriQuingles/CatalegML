# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 12:44:27 2025

@author: ariad
"""
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util

def evaluate_mostra_model(
    mostra_clean_path=r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/Mostra_cleaned.xlsx",
    classif_cleaned_path=r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/Classif_cleaned.xlsx",
    model_path=r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/fine_tuned_model_150_epoca3",
    output_excel_path=r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/evaluation_results150_epoca3.xlsx",
    base_model_name="sentence-transformers/distiluse-base-multilingual-cased-v1"
):
    """
    1) Carrega el model existent desat a 'model_path'.
    2) Llegeix la mostra netejada (Mostra_clean.xlsx) amb 4 columnes: 
       [Codi_HSPAU, Descripcio_HSPAU, Codi_Classif, ICS_Grup_article].
    3) Llegeix Classif_cleaned.xlsx (3 columnes: [Codi_Classif, Grup_Articles, Descripcio_Classif]).
    4) Per cada fila de mostra, calcula la millor coincidència a Classif via cosinus. 
       Afegeix 'Codi_Classif_Pred', 'ICS_Grup_article_Pred' i 'Similitud' al DataFrame.
    5) Desa un nou Excel amb aquesta informació. 
    6) Opcional: Afegeix una columna 'Encert' si coincideix amb el codi ICS real.
    """

    # 1) Carregar model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Carregant el model des de: {model_path}")
    model = SentenceTransformer(model_path, device=device)

    # 2) Llegeix la mostra netejada
    df_mostra = pd.read_excel(mostra_clean_path)
    # Columns: [Codi_HSPAU, Descripcio_HSPAU, Codi_Classif, ICS_Grup_article]

    # 3) Llegeix Classif_cleaned
    df_classif = pd.read_excel(classif_cleaned_path)
    # Assegurem que té [Codi_Classif, Grup_Articles, Descripcio_Classif]
    df_classif.columns = ["Codi_Classif", "Grup_Articles", "Descripcio_Classif"]

    # Genera embeddings de TOT Classif un cop
    classif_embeddings = model.encode(df_classif["Descripcio_Classif"].tolist(), convert_to_tensor=True)

    # Llistes per als resultats
    codi_classif_pred_list = []
    grup_article_pred_list = []
    descripcio_pred_list   = []
    sim_list               = []
    encert_list            = []  # Si vols marcar si coincideix

    for idx, row in df_mostra.iterrows():
        codi_hpau    = row["Codi_HSPAU"]
        desc_hpau    = str(row["Descripcio_HSPAU"])
        codi_classif_real = str(row["Codi_Classif"])

        # Embedding de la descripció HSPAU
        emb_hpau = model.encode(desc_hpau, convert_to_tensor=True)

        # Similitud amb tots els embeddings de Classif
        sims = util.cos_sim(emb_hpau, classif_embeddings)[0]
        max_sim_val, max_idx = torch.max(sims, dim=0)
        max_idx = int(max_idx.item())

        # Recuperar la fila de Classif amb la millor similitud
        best_row = df_classif.iloc[max_idx]
        codi_classif_pred = best_row["Codi_Classif"]
        grup_art_pred     = best_row["Grup_Articles"]
        desc_classif_pred = best_row["Descripcio_Classif"]
        sim_score         = float(max_sim_val.item())

        codi_classif_pred_list.append(codi_classif_pred)
        grup_article_pred_list.append(grup_art_pred)
        descripcio_pred_list.append(desc_classif_pred)
        sim_list.append(sim_score)

        # Opcional: Verificar si coincideix amb el codi ICS real
        encert = (codi_classif_pred == codi_classif_real)
        encert_list.append(encert)

    # Afegir aquestes columnes al df_mostra
    df_mostra["Codi_Classif_Pred"] = codi_classif_pred_list
    df_mostra["ICS_Grup_article_Pred"] = grup_article_pred_list
    df_mostra["Descripcio_Classif_Pred"] = descripcio_pred_list
    df_mostra["Similitud"] = sim_list
    df_mostra["Encert"] = encert_list  # True/False

    # Desa un nou Excel amb resultats
    df_mostra.to_excel(output_excel_path, index=False)
    print(f"Resultats desats a: {output_excel_path}")
    print(f"Total files processades: {len(df_mostra)}")
    # Si vols veure l'accuracy global:
    accuracy = sum(encert_list) / len(encert_list) * 100
    print(f"Accuracy aproximada: {accuracy:.2f}%")

def main():
    mostra_clean_path    = r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/Mostra_cleaned.xlsx"
    classif_cleaned_path = r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/Classif_cleaned.xlsx"
    model_path           = r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/fine_tuned_model_150_epoca3"
    output_excel_path    = r"C:/Users/ariad/Desktop/Cataleg_Arxius_prova/evaluation_results150_epoca3.xlsx"

    evaluate_mostra_model(
        mostra_clean_path=mostra_clean_path,
        classif_cleaned_path=classif_cleaned_path,
        model_path=model_path,
        output_excel_path=output_excel_path,
        base_model_name="sentence-transformers/distiluse-base-multilingual-cased-v1"
    )

if __name__ == "__main__":
    main()

