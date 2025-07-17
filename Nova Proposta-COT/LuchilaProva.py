# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 15:15:56 2025

@author: ariad
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import openai
import time
import re
import logging

client = openai.OpenAI(api_key="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")
logging.basicConfig(filename='clasificacion_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

xls = pd.ExcelFile(r"C:/Users/ariad/Desktop/COT_penjar/Nova Proposta/protesis_cot.xlsx")
regist_df = xls.parse("Total Protesis")
classif_df = xls.parse("Classif")
mostres_df = xls.parse("Mostres Miriam")

for df in [regist_df, classif_df, mostres_df]:
    df.columns = df.columns.str.strip()

labeled_df = mostres_df.merge(classif_df, how="left", left_on="Codi_ICS", right_on="Grup Art.")
labeled_df = labeled_df[['Descripcio_HSPAU', 'Grup Art.', "Grup d'articles", 'Descripcio_Classif']].dropna()

model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
labeled_embeddings = model.encode(labeled_df['Descripcio_HSPAU'].tolist(), convert_to_tensor=True)

def parse_llm_response(text):
    text = text.strip()
    match = re.search(r'([^\n|]+?)\s*\|\s*([^\n|]+?)\s*\|\s*([^\n|]+)', text)
    if match:
        return {
            "Grup Art.": match.group(1).strip(),
            "Grup d'articles": match.group(2).strip(),
            "Descripcio_Classif": match.group(3).strip()
        }
    match2 = re.search(r'\(([^,]+?),\s*([^,]+?),\s*([^)]+)\)', text)
    if match2:
        return {
            "Grup Art.": match2.group(1).strip(),
            "Grup d'articles": match2.group(2).strip(),
            "Descripcio_Classif": match2.group(3).strip()
        }
    match3 = re.findall(r'([A-Z0-9]{8,})\s*\|\s*([^\n|]+?)\s*\|\s*([^\n|]+)', text)
    if match3:
        return {
            "Grup Art.": match3[0][0].strip(),
            "Grup d'articles": match3[0][1].strip(),
            "Descripcio_Classif": match3[0][2].strip()
        }
    raise ValueError(f"Respuesta LLM malformada: {text}")

def classify_with_llm(input_desc, top_k=5):
    input_emb = model.encode(input_desc, convert_to_tensor=True)
    scores = util.cos_sim(input_emb, labeled_embeddings)[0]
    top_idx = torch.topk(scores, k=top_k*2).indices.tolist()

    seen = set()
    examples = []
    for idx in top_idx:
        row = labeled_df.iloc[idx]
        if row['Descripcio_HSPAU'] not in seen:
            seen.add(row['Descripcio_HSPAU'])
            examples.append(row)
        if len(examples) >= top_k:
            break

    prompt = "Eres un experto en clasificación de productos hospitalarios.\n" \
             "Tu tarea es clasificar una descripción según los siguientes ejemplos.\n" \
             "Formato: Grup Art. | Grup d'articles | Descripcio_Classif\n\n" \
             "Ejemplos:\n"

    for row in examples:
        prompt += '- "{}" → {} | {} | {}\n'.format(
            row['Descripcio_HSPAU'],
            row['Grup Art.'],
            row["Grup d'articles"],
            row['Descripcio_Classif']
        )

    prompt += '\nAhora clasifica esto:\n"{}"\n'.format(input_desc)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()

def safe_classify_with_llm(desc, retries=2):
    for attempt in range(retries):
        try:
            return parse_llm_response(classify_with_llm(desc, top_k=5))
        except Exception as e:
            logging.warning(f"LLM intento {attempt+1} fallido para '{desc}': {e}")
            time.sleep(2 ** attempt)

    try:
        return parse_llm_response(classify_with_llm(desc, top_k=10))
    except Exception as e:
        logging.error(f"LLM error persistente para '{desc}': {e}")
        return {
            "Grup Art.": "",
            "Grup d'articles": "",
            "Descripcio_Classif": "NO CLASIFICABLE"
        }

results = []
for _, row in regist_df.iterrows():
    desc = row["Descripcio_HSPAU"]
    try:
        if not isinstance(desc, str) or len(desc.strip().split()) < 2:
            raise ValueError("Descripción no válida.")

        input_emb = model.encode(desc, convert_to_tensor=True)
        scores = util.cos_sim(input_emb, labeled_embeddings)[0]
        best_idx = torch.argmax(scores).item()
        best_score = scores[best_idx].item()
        best_match = labeled_df.iloc[best_idx]

        if best_score >= 0.75:
            source = "embedding"
            result = {
                "Grup Art.": best_match["Grup Art."],
                "Grup d'articles": best_match["Grup d'articles"],
                "Descripcio_Classif": best_match["Descripcio_Classif"]
            }
        else:
            source = "llm"
            result = safe_classify_with_llm(desc)

        results.append({
            "Codi_HSPAU": row["Codi_HSPAU"],
            "Descripcio_HSPAU": desc,
            **result,
            "Similitud": round(best_score, 4),
            "Fuente": source
        })

    except Exception as e:
        logging.error(f"Error clasificando '{desc}': {e}")
        results.append({
            "Codi_HSPAU": row.get("Codi_HSPAU", ""),
            "Descripcio_HSPAU": desc,
            "Grup Art.": "",
            "Grup d'articles": "",
            "Descripcio_Classif": "ERROR",
            "Similitud": 0.0,
            "Fuente": "ERROR"
        })

final_df = pd.DataFrame(results)
final_df.to_excel("clasificacion_hibrida2_TotalProtesis.xlsx", index=False)
final_df[final_df["Descripcio_Classif"].isin(["ERROR", "NO CLASIFICABLE"])].to_excel("errores_clasificacion.xlsx", index=False)

print("Clasificación completada. Resultados guardados en:")
print("- clasificacion_hibrida2_TotalProtesis.xlsx")
print("- errores_clasificacion.xlsx")