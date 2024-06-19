## DATA 

## Import
import config

import os
import json

import re
import string

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

import random
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf

from transformers import AutoTokenizer, TFAutoModel


#region ----- Définir les seed
np.random.seed(config.SEED)
random.seed(config.SEED)
tf.random.set_seed(config.SEED)
#endregion


#region ----- Fonctions utilisées dans ce script

def add_spaces_around_ponctuation(text: str):
    """
    Ajoute des espaces autour de chaque signe de ponctuation dans le texte donné.

    Args:
        text (str): Le texte à modifier.

    Returns:
        str: Le texte avec des espaces ajoutés autour de la ponctuation.
    """
    ponctuations = set(string.punctuation)
    new_text = ""
    for i, char in enumerate(text):
        if char in ponctuations:
            if i > 0 and text[i-1] != " ":
                new_text += " "
            new_text += char
            if i < len(text) - 1 and text[i+1] != " ":
                new_text += " "
        else:
            new_text += char
    return new_text


def create_dataframe(data_folder_path: str, bins: list):
    """
    Crée un DataFrame à partir des fichiers texte dans le dossier spécifié, traite le texte,
    extrait les étiquettes et crée des mappings pour certaines colonnes.

    Args :
        data_folder_path (str) : Chemin vers le dossier contenant les fichiers texte.
        bins (list) : Liste des bords des intervalles pour catégoriser les dates.

    Returns :
    - pd.DataFrame : DataFrame traité avec du texte et des fonctionnalités supplémentaires.
    """
    def create_df(txt_files: str, base_folder: str):
        """
        Crée un DataFrame à partir de fichiers texte dans un dossier donné.

        Args:
            txt_files (list): Liste des noms de fichiers texte.
            base_folder (str): Le chemin du dossier contenant les fichiers texte.

        Returns:
            pd.DataFrame: DataFrame contenant le contenu des fichiers et leurs noms.
        """
        texts = []
        for txt_file in txt_files:
            with open(os.path.join(base_folder, txt_file), "r", encoding="utf-8") as f:
                content = f.read()
            texts.append(content)
        df = pd.DataFrame({
            "text": texts,
            "file_name": txt_files
        })
        return df
    # 1 : Création du DataFrame
    files = remove_invalid_txt_files(get_txt_files(data_folder_path))
    df = create_df(txt_files=files, base_folder=data_folder_path)

    # 2 : Prétraiter le texte
    tqdm.pandas()
    df["text"] = df["text"].progress_apply(add_spaces_around_ponctuation)

    # 3 : Ajout de la colonne intervalle
    df = extract_labels(df=df).sort_values(by="date", ascending=True)
    df["date_interval"] = pd.cut(df["date"], bins=bins, right=False)

    return df


def create_interval(min_value: int, max_value: int, step: int):
    """
    Crée des paires d'intervalles à partir des valeurs min et max avec un pas donné.

    Args:
        min_value (int): La valeur minimale de l'intervalle.
        max_value (int): La valeur maximale de l'intervalle.
        step (int): La taille du pas entre chaque intervalle.

    Returns:
        list: Une liste de paires représentant les intervalles.
    """
    interval_pairs = []
    current_value = min_value
    while current_value <= max_value:
        next_value = current_value + step
        interval_pairs.append([current_value, next_value-1])
        current_value = next_value
    return interval_pairs


def create_mapping(df: pd.DataFrame):
    """
    Crée des mappings à partir d'un DataFrame donné.

    Args:
        df (pd.DataFrame): Le DataFrame source contenant les données.

    Returns:
        dict: Un dictionnaire contenant les mappings créés.
    """
    # Création des maps
    label_date_mapping = {}
    label_sexe_mapping = {
        0: 2, # Femme
        1: 1, # Homme
    }
    
    intervals = df["date_interval"].unique().tolist()
    intervals = sorted(intervals)
    
    for index, interval in enumerate(intervals):
        label_date_mapping[index] = interval
        
    return label_sexe_mapping, label_date_mapping


def extract_labels(df: pd.DataFrame):
    """
    Extrait et ajoute des labels à partir des noms de fichiers dans un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant une colonne "file_name" avec les noms de fichiers.

    Returns:
        pd.DataFrame: Le DataFrame avec des colonnes supplémentaires pour chaque label extrait.
    """
    # Define a lambda function to extract the desired pattern from each file name
    extract_pattern = lambda x: re.findall(r'\((.*?)\)', x)

    # Apply the lambda function to each value in the "file_name" column
    extract_labels = df["file_name"].apply(extract_pattern)
    
    df["nom"] = extract_labels.str[0]
    df["prenom"] = extract_labels.str[1]
    df["titre"] = extract_labels.str[2]
    df["sexe"] = extract_labels.str[3].astype(int)
    df["date"] = extract_labels.str[4].astype(int)

    return df


def generate_attention_mask(input_ids):
    """
    Génère un masque d'attention pour les indices (tokens) d'entrée donnés.

    Args:
        input_ids (tf.Tensor): Un tenseur contenant les indices d'entrée.

    Returns:
        tf.Tensor: Un tenseur de masque d'attention où les positions contenant 1 dans input_ids sont marquées comme 0 et les autres positions comme 1.
    """
    attention_mask = tf.cast(tf.math.logical_not(tf.math.equal(input_ids, 1)), tf.int32)
    return attention_mask


def get_inputs_and_labels_and_file_name(df: pd.DataFrame):
    """
    Extrait les indices (tokens) d'entrée, le masque d'attention, les étiquettes de sexe et de date, et les noms de fichiers d'un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les colonnes "input_ids", "attention_mask", "sexe", "date" et "file_name".

    Returns:
        tuple: Un tuple contenant :
            - Une liste avec deux tf.Tensor : les identifiants d'entrée et le masque d'attention.
            - tf.Tensor : les étiquettes de sexe.
            - tf.Tensor : les étiquettes de date.
            - tf.Tensor : les noms de fichiers.
    """
    inputs_ids = tf.stack([tf.squeeze(tensor, axis=0) for tensor in df["input_ids"].tolist()], axis=0)
    attention_mask = tf.stack([tf.squeeze(tensor, axis=0) for tensor in df["attention_mask"].tolist()], axis=0)
    sexe_labels = tf.convert_to_tensor(df["sexe"].tolist(), dtype=tf.float32)
    date_labels = tf.convert_to_tensor(df["date"].tolist(), dtype=tf.float32)
    file_name = tf.convert_to_tensor(df["file_name"].tolist(), dtype=tf.string)

    return [inputs_ids, attention_mask], sexe_labels, date_labels, file_name


def get_txt_files(directory: str, extension: str=".txt"):
    """
    Obtient les noms de tous les fichiers texte dans un répertoire donné.

    Args:
        directory (str): Le chemin vers le répertoire.
        extension (str, optional): L'extension des fichiers à rechercher. Par défaut ".txt".

    Returns:
        list: Une liste contenant les noms de tous les fichiers texte dans le répertoire.
    """
    # List all files in the directory
    all_files = os.listdir(directory)

    # Filter out the text files
    txt_files = [file for file in all_files if file.endswith(extension)]

    return txt_files


def information(df: pd.DataFrame, folder_path: str, prefix: str, y_axis_label: str = "Nombre de textes"):
    """
    Génère des graphiques et fournit des statistiques descriptives pour un DataFrame.
    
    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
        folder_path (str): Le chemin où les graphiques seront enregistrés.
        prefix (str): Le préfixe pour les noms de fichiers des graphiques.
        y_axis_label (str, optional): Le label pour l'axe Y. Par défaut "Nombre de textes".
        
    Returns:
        dict: Un dictionnaire contenant les statistiques descriptives et les comptages.
    """
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.countplot(x="date_interval", hue="sexe", data=df, palette="tab10")
    plt.title("Répartition combinée par sexe et intervalles (25 ans)")
    plt.xlabel("Intervalle")
    plt.ylabel(y_axis_label)
    plt.xticks(rotation=45)
    plt.legend(title="sexe", labels=["Homme", "Femme"])
    plt.tight_layout()
    plt.grid(True)
    
    plt.savefig(os.path.join(folder_path, f"{prefix}_histogramme.png"))
    
    plt.close() 
    
    
    # Dictionnaire pour les étiquettes
    labels = {1: "Homme", 2: "Femme"}

    # Comptage des valeurs
    counts = df["sexe"].value_counts()

    # Remplacement des valeurs par les étiquettes pour l'affichage
    label_values = [labels[key] for key in counts.index]

    # Création du graphique camembert
    plt.figure(figsize=(8, 8))
    plt.pie(counts, labels=label_values, autopct="%1.1f%%", startangle=140)
    plt.title("Répartition des sexes")
    plt.axis("equal")  # Aspect ratio égal pour que le camembert soit dessiné comme un cercle
    
    plt.savefig(os.path.join(folder_path, f"{prefix}_camembert.png"))
    
    plt.close() 
    
    result = {}
    
    # Describe
    result["value_counts_date_interval"] = df["date_interval"].value_counts().to_dict()
    result["descriptive_stats_date_interval"] = df["date_interval"].describe().to_dict()
    result["value_counts_sexe"] = df["sexe"].value_counts().to_dict()
    
    # Groupby and count
    counts = df.groupby(["date_interval", "sexe"], observed=False).size().reset_index(name="count")
    result["unique_combinations_counts"] = counts.to_dict(orient="records")
    result["descriptive_stats_counts"] = counts["count"].describe().to_dict()
    
    return result


def load_labels(data: pd.DataFrame, date_interval: bool=False):
    """
    Charge les étiquettes à partir du DataFrame.

    Args:
        data (pd.DataFrame): Le DataFrame contenant les étiquettes.
        date_interval (bool, optional): Indique s'il faut charger les étiquettes de date ou d'intervalle de date. Par défaut False.

    Returns:
        tuple: Un tuple contenant les étiquettes pour le sexe et la date/interval de date.
    """
    if not date_interval:
        date_labels = tf.convert_to_tensor(data["date"].tolist(), dtype=tf.float32)
    else:
        date_labels = tf.convert_to_tensor(data["date_interval"].tolist(), dtype=tf.float32)
    
    sexe_labels = tf.convert_to_tensor(data["sexe"].tolist(), dtype=tf.float32)
    
    return sexe_labels, date_labels


def oversample_df(df):
    """
    Suréchantillonne le DataFrame pour équilibrer les classes.

    Args:
        df (pd.DataFrame): Le DataFrame à suréchantillonner.

    Returns:
        pd.DataFrame: Le DataFrame suréchantillonné avec des classes équilibrées.
    """
    df["combine_interval_sexe"] = df["date_interval"].astype(str) + "-" + df["sexe"].astype(str)
    df = df.sample(frac=1).reset_index(drop=True)
    min_count = df["combine_interval_sexe"].value_counts().min()
    # Oversample each value to have the same number of rows as the minimum count
    df = df.groupby("combine_interval_sexe").apply(lambda x: x.sample(min_count, replace=True))
    # Reset the index
    df = df.reset_index(drop=True)
    
    return df.drop("combine_interval_sexe", axis=1)


def process_dataframe(df: pd.DataFrame, max_slice_length: int=514):
    """
    Tokenise le texte dans le DataFrame, découpe et remplit les tensors résultants,
    et retourne une liste de DataFrames segmentés.

    Args:
        df (pd.DataFrame): DataFrame contenant les données à traiter.
        tokenizer: Tokenizer pour tokeniser le texte.
        max_slice_length (int): Taille maximale d'un morceau.

    Returns:
        list of pd.DataFrame: Liste des DataFrames segmentés.
    """
    # Tokeniser tout le DataFrame à la fois
    
    dfs = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        input_ids = split_input_ids(
            input_ids=row["text_tensor"],
            max_slice_length=max_slice_length
        )
        segment_df = pd.DataFrame({
            "input_ids": input_ids,
        })
        
        # Copy other columns to each new DataFrame
        for col in df.columns:
            if col not in ["text", "text_tensor"] :
                segment_df[col] = row[col]
        
        dfs.append(segment_df)
    
    return pd.concat(dfs, ignore_index=True) 


def remove_invalid_txt_files(txt_files: list):
    """
    Supprime les fichiers texte invalides en fonction de critères spécifiques.

    Args:
        txt_files (list): Liste des noms de fichiers texte à filtrer.

    Returns:
        list: Liste des noms de fichiers texte valides après filtrage.
    """
    valid_files = []
    invalid_files = []
    for file_name in txt_files:
        # Extraire les éléments entre parenthèses
        labels = re.findall(r"\((.*?)\)", file_name)
        
        if len(labels) >= 5:
            # Vérifier si l"élément à l'indice 3 est égal à 1 ou 2
            if labels[3] in ['1', '2']:
                # Vérifier si l'élément à l"indice 4 est un entier
                if labels[4].isdigit():
                    # Ajouter le fichier valide à la liste des fichiers valides
                    valid_files.append(file_name)
                else:
                    # Ajouter le nom de fichier non retenu à la liste des fichiers non valides
                    invalid_files.append(file_name)
            else:
                # Ajouter le nom de fichier non retenu à la liste des fichiers non valides
                invalid_files.append(file_name)
        else:
            # Ajouter le nom de fichier non retenu à la liste des fichiers non valides
            invalid_files.append(file_name)
    
    # Afficher les noms de fichiers non retenus
    for file_name in invalid_files:
        print("Fichier non retenu:", file_name)
    
    return valid_files


def split_dataframe(df: pd.DataFrame, size: float, random_state=None):
    """
    Divise le DataFrame en deux parties selon une taille spécifiée.

    Args:
        df (pd.DataFrame): Le DataFrame à diviser.
        size (float): La proportion de lignes à garder dans la partie 'keep'.
        random_state (int or None, optional): Contrôle la reproductibilité de l'échantillonnage aléatoire. Par défaut None.

    Returns:
        pd.DataFrame: Le DataFrame contenant les lignes conservées après division.
    """
    # Create a new column that combines the values of "date_interval" and "sexe"
    df["stratify_column"] = df["date_interval"].astype(str) + "_" + df["sexe"].astype(str)
    
    if size == 1.0:
        return df.sort_index().reset_index(drop=True)  # Return the entire DataFrame
    
    # Split into keep and don't keep
    _, keep = train_test_split(df, test_size=size, stratify=df["stratify_column"], random_state=random_state)
    
    return keep.sort_index().reset_index(drop=True)


def split_dataframe_into_train_and_val(df: pd.DataFrame, train_size: float):
    """
    Divise un DataFrame en ensembles d'entraînement et de validation en fonction d'une taille spécifiée.

    Args:
        df (pd.DataFrame): Le DataFrame à diviser.
        train_size (float): La proportion des données à inclure dans l'ensemble d'entraînement.

    Returns:
        tuple: Un tuple contenant deux DataFrames, le premier pour l'ensemble d'entraînement et le second pour l'ensemble de validation.
    """

    # Group by 'sexe' and 'interval_date'
    grouped = df.groupby(["sexe", "date_interval"])

    # Create two empty dataframes
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()

    # Iterate over groups
    for group_key, group_df in grouped:
        # Split the group into two dataframes based on the rate
        df1_group, df2_group = group_df.sample(frac=1).reset_index(drop=True).iloc[:int(len(group_df) * train_size)], \
                               group_df.sample(frac=1).reset_index(drop=True).iloc[int(len(group_df) * train_size):]

        # Concatenate to the overall result dataframes
        df1 = pd.concat([df1, df1_group])
        df2 = pd.concat([df2, df2_group])

    return df1, df2


def split_input_ids(input_ids, max_slice_length: int=514):
    """
    Découpe un tensor d'identifiants d'entrée en morceaux de longueur maximale spécifiée.

    Args:
        input_ids (tf.Tensor): Le tensor d'identifiants d'entrée à découper.
        max_slice_length (int, optional): La longueur maximale de chaque morceau. Par défaut 514.

    Returns:
        list: Une liste de tensors, chaque tensor étant un morceau de input_ids avec padding ajouté si nécessaire.
    """
    size = tf.shape(input_ids)[1]

    # Calculer le nombre de morceaux nécessaires
    num_slices = size // (max_slice_length - 2) + int(size % (max_slice_length - 2) != 0)

    # Découper le tensor en morceaux
    input_ids_slices = []
    for i in range(num_slices):
        start = i * (max_slice_length - 2)
        end = min((i + 1) * (max_slice_length - 2), size)
        slice_length = end - start
        
        # Ajouter 5 au début et 6 à la fin
        input_ids_slice = tf.concat([tf.constant([[5]], dtype=input_ids.dtype), input_ids[:, start:end], tf.constant([[6]], dtype=input_ids.dtype)], axis=1)
        
        # Padding si la longueur du morceau est inférieure à max_slice_length
        if slice_length + 2 < max_slice_length:
            padding_length = max_slice_length - (slice_length + 2)
            input_ids_slice = tf.pad(input_ids_slice, [[0, 0], [0, padding_length]], constant_values=1)

        input_ids_slices.append(input_ids_slice)

    return input_ids_slices


def tokenize_text(text: str, tokenizer):
    """
    Tokenize un texte en utilisant un tokenizer donné.

    Args:
        text (str): Le texte à tokenizer.
        tokenizer: Un objet tokenizer capable de tokenizer le texte.

    Returns:
        tf.Tensor: Le tensor contenant les identifiants d'entrée tokenizés du texte.
    """
    # Découper le texte en parties selon les points suivis d'un espace
    text_parts = re.split(r'(?<=\.)\s', text)

    # Initialisation des tenseurs vides pour les parties tokenizées
    tokenized_parts = []

    # Tokenizer individuellement chaque partie et collecter les tenseurs
    for part in text_parts:
        tokenized_part = tokenizer(part, padding=False, truncation=False, return_tensors="tf")
        tokenized_parts.append(tokenized_part)

    # Concaténer les parties tokenizées
    concatenated_tokens = {}
    for key in tokenized_parts[0].keys():
        concatenated_tokens[key] = tf.concat([part[key] for part in tokenized_parts], axis=1)

    input_ids_tensor = concatenated_tokens["input_ids"]

    # Filtrer les valeurs 5 et 6
    filtered_tensor = tf.boolean_mask(input_ids_tensor, tf.logical_and(input_ids_tensor != 5, input_ids_tensor != 6))

    # Ajouter 5 au début et 6 à la fin
    # tensor_with_boundaries = tf.concat([[5], filtered_tensor, [6]], axis=0)

    # Reshape pour maintenir la forme originale (1, -1)
    final_tensor = tf.reshape(filtered_tensor, [1, -1])

    return final_tensor

#endregion


def main():
    
    print("\nDonnées...\n")
    
    tqdm.pandas()
    
    # Utilise des fichiers _balanced.pkl s'ils existent
    if config.USE_DATA_BALANCED \
        and os.path.exists(config.TRAIN_BALANCED_PKL_PATH) \
        and os.path.exists(config.TEST_REDUCED_PKL_PATH) \
        and os.path.exists(config.VAL_BALANCED_PKL_PATH):

        print("\nChargement _balanced.pkl...\n")

        train_balanced_df = pd.read_pickle(config.TRAIN_BALANCED_PKL_PATH)
        test_reduced_df = pd.read_pickle(config.TEST_REDUCED_PKL_PATH)
        val_balanced_df = pd.read_pickle(config.VAL_BALANCED_PKL_PATH)
    
    elif not config.USE_DATA_BALANCED \
        and os.path.exists(config.TRAIN_REDUCED_PKL_PATH) \
        and os.path.exists(config.TEST_REDUCED_PKL_PATH) \
        and os.path.exists(config.VAL_REDUCED_PKL_PATH):
        
        print("\nChargement _reduced.pkl...\n")
        
        train_reduced_df = pd.read_pickle(config.TRAIN_REDUCED_PKL_PATH)
        test_reduced_df = pd.read_pickle(config.TEST_REDUCED_PKL_PATH)
        val_reduced_df = pd.read_pickle(config.VAL_REDUCED_PKL_PATH)
        
    else:
        # Création des fichiers .pkl s'ils n'existent pas
        if not os.path.exists(config.TRAIN_PKL_PATH) \
            and not os.path.exists(config.TEST_PKL_PATH) \
            and not os.path.exists(config.VAL_PKL_PATH):
        
            print("\nCréation .pkl...\n")
        
            # 1 : Création des intervalles de 25 ans 
            # intervals = create_interval(min_value=1825, max_value=2023, step=25)
            intervals = [[int(start), int(end) - 1] for interval in list(config.DATE_MAP.values()) for start, end in [interval.strip("[]()").split(", ")]]
            bins = [interval[0] for interval in intervals] + [intervals[-1][1]]
            # bins = [interval[0] for interval in list(config.DATE_MAP.values())] + [list(config.DATE_MAP.values())[-1][1]]

            # 2 : création des DataFrame
            print("\nPré traitement des données & création des dataframes...\n")
            train_df = create_dataframe(config.TRAIN_DATA_PATH, bins).reset_index(drop=True)
            test_df = create_dataframe(config.TEST_DATA_PATH, bins).reset_index(drop=True)
            val_df = create_dataframe(config.VAL_DATA_PATH, bins).reset_index(drop=True)

            assert not train_df.columns.duplicated().any(), "Duplications dans les colonnes de train_df"
            assert not train_df.index.duplicated().any(), "Duplications dans l'index de train_df"
            assert not test_df.columns.duplicated().any(), "Duplications dans les colonnes de test_df"
            assert not test_df.index.duplicated().any(), "Duplications dans l'index de test_df"
            assert not val_df.columns.duplicated().any(), "Duplications dans les colonnes de val_df"
            assert not val_df.index.duplicated().any(), "Duplications dans l'index de val_df"


            # 0.1 : Information des DataFrames
            _ = information(df=train_df, folder_path=config.FIGURES_PATH, prefix="train", y_axis_label="Nombre de textes")
            _ = information(df=test_df, folder_path=config.FIGURES_PATH, prefix="test", y_axis_label="Nombre de textes")
            _ = information(df=val_df, folder_path=config.FIGURES_PATH, prefix="val",y_axis_label="Nombre de textes")

            df = pd.concat([train_df, test_df, val_df]).reset_index(drop=True)
            assert not df.columns.duplicated().any(), "Duplications dans les colonnes de df"
            assert not df.index.duplicated().any(), "Duplications dans l'index de df"
            _ = information(df=df, folder_path=config.FIGURES_PATH, prefix="global", y_axis_label="Nombre de textes")

            # 3 : Tokenization des textes
            tokenizer = AutoTokenizer.from_pretrained("cmarkea/distilcamembert-base")

            print("\nTokenization...\n")
            train_df["text_tensor"] = train_df["text"].progress_apply(lambda text: tokenize_text(text, tokenizer=tokenizer))
            test_df["text_tensor"] = test_df["text"].progress_apply(lambda text: tokenize_text(text, tokenizer=tokenizer))
            val_df["text_tensor"] = val_df["text"].progress_apply(lambda text: tokenize_text(text, tokenizer=tokenizer))

            # 3.1 : Division des tensors
            print("\nCreation des inputs ids...\n")
            train_df = process_dataframe(train_df)
            test_df = process_dataframe(test_df)
            val_df = process_dataframe(val_df)

            # 3.2 : Generation des masques d'attentions
            print("\nGeneration des attention mask...\n")
            train_df["attention_mask"] = train_df["input_ids"].progress_apply(generate_attention_mask)
            test_df["attention_mask"] = test_df["input_ids"].progress_apply(generate_attention_mask)
            val_df["attention_mask"] = val_df["input_ids"].progress_apply(generate_attention_mask)

            # 0.2 : Enregistrement des DataFrames
            train_df.to_pickle(config.TRAIN_PKL_PATH)
            test_df.to_pickle(config.TEST_PKL_PATH)
            val_df.to_pickle(config.VAL_PKL_PATH)
        
        else:
            print("\nChargement .pkl...\n")
            
            train_df = pd.read_pickle(config.TRAIN_PKL_PATH)
            test_df = pd.read_pickle(config.TEST_PKL_PATH)
            val_df = pd.read_pickle(config.VAL_PKL_PATH)
            
            # 0.1 : Information des DataFrames
            _ = information(df=train_df, folder_path=config.FIGURES_PATH, prefix="train", y_axis_label="Nombre de textes")
            _ = information(df=test_df, folder_path=config.FIGURES_PATH, prefix="test", y_axis_label="Nombre de textes")
            _ = information(df=val_df, folder_path=config.FIGURES_PATH, prefix="val",y_axis_label="Nombre de textes")
            
            df = pd.concat([train_df, test_df, val_df]).reset_index(drop=True)
            _ = information(df=df, folder_path=config.FIGURES_PATH, prefix="global", y_axis_label="Nombre de textes")    

        
        # 4 : Mappage des DataFrames
        train_df["date_interval"] = train_df["date_interval"].astype(str)
        test_df["date_interval"] = test_df["date_interval"].astype(str)
        val_df["date_interval"] = val_df["date_interval"].astype(str)
        df["date_interval"] = df["date_interval"].astype(str)
        # sexe_mapping, date_mapping = create_mapping(df=df)
        sexe_mapping, date_mapping = config.SEXE_MAP, config.DATE_MAP

        _ = information(df=train_df, folder_path=config.FIGURES_PATH, prefix="train_segment", y_axis_label="Nombre de segments")
        _ = information(df=test_df, folder_path=config.FIGURES_PATH, prefix="test_segment", y_axis_label="Nombre de segments")
        _ = information(df=val_df, folder_path=config.FIGURES_PATH, prefix="val_segment",y_axis_label="Nombre de segments")

        train_df["sexe"] = train_df["sexe"].map(dict(map(reversed, sexe_mapping.items())))
        test_df["sexe"] = test_df["sexe"].map(dict(map(reversed, sexe_mapping.items())))
        val_df["sexe"] = val_df["sexe"].map(dict(map(reversed, sexe_mapping.items())))

        train_df["date_interval"] = train_df["date_interval"].map(dict(map(reversed, date_mapping.items())))
        test_df["date_interval"] = test_df["date_interval"].map(dict(map(reversed, date_mapping.items())))
        val_df["date_interval"] = val_df["date_interval"].map(dict(map(reversed, date_mapping.items())))
        
        if config.USE_DATA_BALANCED:
            print("\nCréation _balanced.pkl...\n")
            
            print(f"\nTaille de l'ensemble train_df: {len(train_df)}")
            print(f"Taille de l'ensemble test_df: {len(test_df)}")
            print(f"Taille de l'ensemble val_df df: {len(val_df)}\n")
            
            concat_df = pd.concat([train_df, val_df]).reset_index(drop=True)
            # print(f"Taille de l'ensemble concat df: {len(concat_df)}\n")
            # _ = information(df=concat_df, folder_path=config.FIGURES_PATH, prefix="concat_segment", y_axis_label="Nombre de segments")
            
            # : Equilibre la quantité de données de chaque classe
            concat_balanced_df = oversample_df(df=concat_df)
            
            train_balanced_df, val_balanced_df = split_dataframe_into_train_and_val(concat_balanced_df, 0.8)
            
            print(f"Taille de l'ensemble concat_df: {len(concat_df)}\n")
            print(f"\nTaille de l'ensemble train_balanced_df: {len(train_balanced_df)}")
            print(f"Taille de l'ensemble val_balanced_df: {len(val_balanced_df)}")
            print(f"Taille de l'ensemble concat_balanced_df: {len(concat_balanced_df)}\n")
            
            # 5 : Réduit la quantité de données dans les 3 ensembles train, test et val 
            train_balanced_df = split_dataframe(df=train_balanced_df, size=config.TRAIN_BALANCED_SIZE, random_state=config.SEED)
            val_balanced_df = split_dataframe(df=val_balanced_df, size=config.VAL_BALANCED_SIZE, random_state=config.SEED)

            print(f"\nTaille de l'ensemble train_balanced_df: {len(train_balanced_df)}")
            print(f"Taille de l'ensemble val_balanced_df: {len(val_balanced_df)}")
            print(f"Taille de l'ensemble concat df: {len(concat_balanced_df)}\n")

            # 0.3 : Information
            _ = information(df=train_balanced_df[["date_interval", "sexe"]].apply(lambda x: x.map(date_mapping) if x.name == "date_interval" else x.map(sexe_mapping)), folder_path=config.FIGURES_PATH, prefix="train_balanced", y_axis_label="Nombre de segments")
            _ = information(df=val_balanced_df[["date_interval", "sexe"]].apply(lambda x: x.map(date_mapping) if x.name == "date_interval" else x.map(sexe_mapping)), folder_path=config.FIGURES_PATH, prefix="val_balanced", y_axis_label="Nombre de segments")
            
            # 0.4 : Enregistrement des DataFrames réduits
            train_balanced_df.to_pickle(config.TRAIN_BALANCED_PKL_PATH)
            val_balanced_df.to_pickle(config.VAL_BALANCED_PKL_PATH)
    
        else:
            print("\nCréation _reduced.pkl...\n")
            
            # 5 : Réduit la quantité de données dans les 3 ensembles train, test et val 
            train_reduced_df = split_dataframe(df=train_df, size=config.TRAIN_REDUCED_SIZE, random_state=config.SEED)
            val_reduced_df = split_dataframe(df=val_df, size=config.VAL_REDUCED_SIZE, random_state=config.SEED)

            # 0.3 : Information
            _ = information(df=train_reduced_df[["date_interval", "sexe"]].apply(lambda x: x.map(date_mapping) if x.name == "date_interval" else x.map(sexe_mapping)), folder_path=config.FIGURES_PATH, prefix="train_reduced", y_axis_label="Nombre de segments")
            _ = information(df=val_reduced_df[["date_interval", "sexe"]].apply(lambda x: x.map(date_mapping) if x.name == "date_interval" else x.map(sexe_mapping)), folder_path=config.FIGURES_PATH, prefix="val_reduced", y_axis_label="Nombre de segments")
            
            # 0.4 : Enregistrement des DataFrames réduits
            train_reduced_df.to_pickle(config.TRAIN_REDUCED_PKL_PATH)
            val_reduced_df.to_pickle(config.VAL_REDUCED_PKL_PATH)

        test_reduced_df = split_dataframe(df=test_df, size=config.TEST_REDUCED_SIZE, random_state=config.SEED)
        _ = information(df=test_reduced_df[["date_interval", "sexe"]].apply(lambda x: x.map(date_mapping) if x.name == "date_interval" else x.map(sexe_mapping)), folder_path=config.FIGURES_PATH, prefix="test_reduced", y_axis_label="Nombre de segments")
        test_reduced_df.to_pickle(config.TEST_REDUCED_PKL_PATH)
            

    if config.USE_DATA_BALANCED:
        print("\nUtilisation _balanced.pkl...\n")
        
        # 6 : Listes contenants les données pour l'entraînement des modèles
        train_inputs, train_sexe_labels, train_date_labels, _ = get_inputs_and_labels_and_file_name(df=train_balanced_df)
        val_inputs, val_sexe_labels, val_date_labels, _ = get_inputs_and_labels_and_file_name(df=val_balanced_df)
    
    else:
        print("\nUtilisation _reduced.pkl...\n")
        
        # 6 : Listes contenants les données pour l'entraînement des modèles
        train_inputs, train_sexe_labels, train_date_labels, _ = get_inputs_and_labels_and_file_name(df=train_reduced_df)
        val_inputs, val_sexe_labels, val_date_labels, _ = get_inputs_and_labels_and_file_name(df=val_reduced_df)

    test_inputs, test_sexe_labels, test_date_labels, test_file_name = get_inputs_and_labels_and_file_name(df=test_reduced_df)

    return {
        "train_inputs": train_inputs,
        "train_sexe_labels": train_sexe_labels,
        "train_date_labels": train_date_labels,
        
        "test_inputs": test_inputs,
        "test_sexe_labels": test_sexe_labels,
        "test_date_labels": test_date_labels,
        "test_file_name": test_file_name,
        
        "val_inputs": val_inputs,
        "val_sexe_labels": val_sexe_labels,
        "val_date_labels": val_date_labels
    }

    
if __name__ == '__main__':
    main()