## CONFIGURATION

#region ----- A ne pas modifier

# Configuration des PATHS
DATA_PATH = "../data/"
TRAIN_DATA_PATH = DATA_PATH + "corpus/train/"
TEST_DATA_PATH = DATA_PATH + "corpus/test/"
VAL_DATA_PATH = DATA_PATH + "corpus/val/"

## Results
RESULTS_PATH = "../results/"
FIGURES_PATH = RESULTS_PATH + "figures/"
TRAINING_PATH = RESULTS_PATH + "training/"

# Nom des fichiers
PKL_PATH = DATA_PATH + "pkl/"
## Fichier d'origine contenant les inputs_ids, etc
TRAIN_PKL_PATH = PKL_PATH + "train.pkl"
TEST_PKL_PATH = PKL_PATH + "test.pkl"
VAL_PKL_PATH = PKL_PATH + "val.pkl"
## Fichiers utilisés durant l'entraînement et l'évaluation
### Répartitions inégales dans les classes (plus de données utilisables)
TRAIN_REDUCED_PKL_PATH = PKL_PATH + "train_reduced.pkl"
TEST_REDUCED_PKL_PATH = PKL_PATH + "test_reduced.pkl"
VAL_REDUCED_PKL_PATH = PKL_PATH + "val_reduced.pkl"
### Répartitions égales de toutes les classes (moins de données utilisables)
TRAIN_BALANCED_PKL_PATH = PKL_PATH + "train_balanced.pkl"
VAL_BALANCED_PKL_PATH = PKL_PATH + "val_balanced.pkl"

# Modèles
MODEL_BASE_FOLDER = "../models/"
MODEL_ID = "cmarkea/distilcamembert-base"

## Modèle multitask, multihead, cnn: MM_cnn
MM_CNN = "mm_cnn"
MM_CNN_RESULT_PATH = RESULTS_PATH + MM_CNN + "/"

## Modèle singletask, singhead, cnn: SS_cnn
SS_CNN = "ss_cnn"
SS_CNN_RESULT_PATH = RESULTS_PATH + SS_CNN + "/"

#endregion


# ----------------------------------------------------------------


#region ----- Modifiable, faire tout de même attention

# Paramètres
## Configuration générale
SEED = 44

## Donnée
### Taille des sous ensembles train, test et val
TRAIN_REDUCED_SIZE = 1 # 0 < size <= 1.0
TEST_REDUCED_SIZE = 1 # 0 < size <= 1.0 # A utiliser dans tous les cas
VAL_REDUCED_SIZE = 0.5 # 0 < size <= 1.0
### Utiliser des classes équilibré ou non pour les ensembles train et val ?
USE_DATA_BALANCED = True
### Taille des sous ensembles train et val équilibrés
TRAIN_BALANCED_SIZE = 0.5 # 0 < size <= 1.0
VAL_BALANCED_SIZE = 0.5 # 0 < size <= 1.0

## Entraînement
### Ajuster ces valeurs en fonction de l'importance relative de chaque composante
DATE_LOSS_WEIGHT = 0.0001  # Plus petite valeur car la perte de date est beaucoup plus grande
SEXE_LOSS_WEIGHT = 1.0

LEARNING_RATE = 1e-5

BATCH_SIZE = 16 
EPOCHS = 20
PATIENCE = 3

## Maps
DATE_MAP = {
    0: "[1825, 1850)", # [inclus, exclu)
    1: "[1850, 1875)", 
    2: "[1875, 1900)", 
    3: "[1900, 1925)", 
    4: "[1925, 1950)", 
    5: "[1950, 1975)", 
    6: "[1975, 2000)", 
    7: "[2000, 2025)"
}
SEXE_MAP = {
    0: 2, # Femme 
    1: 1  # Homme
}

#endregion

