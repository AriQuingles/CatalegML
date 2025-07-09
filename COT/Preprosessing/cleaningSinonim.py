# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 22:33:41 2025

@author: AQuingles
"""
import pandas as pd
import re
import unicodedata

#############################################
# 1) Llegir Excel de sinònims i construir-ne un diccionari
#############################################
def build_synonyms_dict(path_dicc):
    """
    Llegeix l'Excel del diccionari (ex: 'Diccionari Protesis.xlsx'),
    on hi ha mínim dues columnes:
      - Col 0: vocabulari ICS
      - Col 1: sinònims HSPAU, separats per comes
    Retorna synonyms_map = {ICS_term: {var1, var2, ...}}
    """
    df_dicc = pd.read_excel(path_dicc)
    
    synonyms_map = {}
    for idx, row in df_dicc.iterrows():
        ics_term = str(row.iloc[0]).strip()
        raw_syns = str(row.iloc[1]).strip()  # pot contenir varis sinònims separats per comes
        variants = [v.strip() for v in raw_syns.split(',')] if raw_syns else []
        
        if ics_term not in synonyms_map:
            synonyms_map[ics_term] = set()
        for var in variants:
            if var:
                synonyms_map[ics_term].add(var)
    
    return synonyms_map


def build_synonyms_regex(synonyms_map: dict):
    """
    Construeix un patró regex i un diccionari invertit 
    a partir de synonyms_map = {ICS_term: {var1, var2, ...}}.
    Retorna:
       pattern, inverse_map
    """
    inverse_map = {}
    for ics_term, variants in synonyms_map.items():
        for var in variants:
            var_clean = var.strip()
            if var_clean:
                # Al diccionari invertit, la clau és el sinònim en minúscules,
                # i el valor és el terme ICS
                inverse_map[var_clean.lower()] = ics_term

    # Llista de sinònims en minúscules (ex: ['torn','cemen','ciment','primaria'...])
    all_variants = list(inverse_map.keys())

    # Patró => \b(sinònim1|sinònim2|...)\b
    pattern = r"\b(" + "|".join(map(re.escape, all_variants)) + r")\b"

    return pattern, inverse_map


def apply_synonyms(text: str, pattern: str, inverse_map: dict) -> str:
    """
    Usa un sol re.sub per substituir sinònims via callback.
    - text: el text original
    - pattern: regex generat per build_synonyms_regex
    - inverse_map: diccionari {sinonim_lower: ics_term}
    """
    def replacement_func(match):
        found = match.group(1)  # El sinònim trobat
        found_lower = found.lower()
        # Retorna la paraula ICS
        return inverse_map[found_lower]

    # IGNORECASE per ignorar majúsc./minúsc.
    new_text = re.sub(pattern, replacement_func, text, flags=re.IGNORECASE)
    return new_text


#########################################
# 2) Processar l'Excel de Classif
#########################################
def process_classif(path_classif):
    """
    Llegeix l'Excel Classif,
    i crea un DataFrame amb les columnes:
      Codi_Classif, ICS_Grup_article, Descripcio_Classif
    No fem cap conversió a minúscules ni eliminem accents.
    """
    df_classif = pd.read_excel(path_classif)
    df_classif.columns = ['Codi_Classif','ICS_Grup_article','Descripcio_Classif']
    return df_classif


#########################################
# 3) Processar l'Excel de Regist
#########################################
def process_regist(path_regist, synonyms_map):
    """
    Llegeix l'Excel de Regist,
    fa servir build_synonyms_regex(synonyms_map) per construir:
      - pattern
      - inverse_map
    i després fa apply_synonyms a la columna 'Descripcio_HSPAU' 3 cops.
    """
    # 1) Crear un sol pattern i map invertit
    pattern, inv_map = build_synonyms_regex(synonyms_map)

    # 2) Llegir l'excel
    df_regist = pd.read_excel(path_regist)
    df_regist.columns = ["Codi_HSPAU", "Descripcio_HSPAU"]

    # 3) Substituir sinònims 3 cops
    for _ in range(3):
        df_regist["Descripcio_HSPAU"] = df_regist["Descripcio_HSPAU"].astype(str).apply(
            lambda x: apply_synonyms(x, pattern, inv_map)
        )

    return df_regist


#########################################
# 4) Processar l'Excel de mostra (opcional)
#########################################
def process_mostra(path_mostra):
    """
    Llegeix l'Excel de mostra (100 materials).
    Assumeix que hi ha almenys 'Descripcio_HSPAU' i 'Codi_Classif', etc.
    """
    df_mostra = pd.read_excel(path_mostra)
    return df_mostra


#########################################
# 5) Main (exemple de flux)
#########################################
def main():
    path_classif = r'C:/Users/ariad/Desktop/Cataleg_Arxius_prova/classif_maj_selecció.xlsx'
    path_regist  = r'C:/Users/ariad/Desktop/Cataleg_Arxius_prova/regist.xlsx'
    path_mostra  = r'C:/Users/ariad/Desktop/Cataleg_Arxius_prova/miriam_mostra_100.xlsx'
    path_dicc    = r'C:/Users/ariad/Desktop/Cataleg_Arxius_prova/Diccionari Protesis.xlsx'
    
    # 1) Construir el diccionari de sinònims a partir de l'Excel
    synonyms_map = build_synonyms_dict(path_dicc)
    
    # 2) Carregar Classif
    df_classif = process_classif(path_classif)
    df_classif.to_excel("Classif_cleaned.xlsx", index=False)
    
    # 3) Carregar Regist i aplicar sinònims (3 passades)
    df_regist = process_regist(path_regist, synonyms_map)
    df_regist.to_excel("Regist_cleaned.xlsx", index=False)
    
    # 4) Carregar mostra (opcional)
    df_mostra = process_mostra(path_mostra)
    df_mostra.to_excel("Mostra_cleaned.xlsx", index=False)
    
    print("Arxius cleaned generats: Classif_cleaned.xlsx, Regist_cleaned.xlsx, Mostra_cleaned.xlsx")


if __name__ == "__main__":
    main()
