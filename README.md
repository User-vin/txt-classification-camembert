



# Introduction
Utilisation de DistilCamemBERT pour la classification de textes romanesques français.

## Objectif du Projet
L'objectif principal de ce projet est d'améliorer les performances de la classification du sexe de l'auteur. Pour cela, entre en jeu un modèle multi-têtes multitâches qui prend également en compte la prédiction de la date de parution du texte. Ce projet vise à déterminer si cette approche permet effectivement une amélioration des performances de la tâche de classification du sexe de l'auteur.


Dans la méthode développée, l'utilisation de modèles de type BERT, en particulier CamemBERT adapté pour le français, sert de base pour l'architecture du modèle.

## Contexte
Ce projet s'inscrit dans un contexte plus large de recherche visant à développer des outils suffisamment performants pour l'analyse de textes romanesques français et l'extraction d'informations telles que le sexe de l'auteur, la date de parution, etc. Des recherches sont notamment menées pour améliorer la précision et l'efficacité de ces outils d'analyse textuelle.

# Installation
## Prérequis
* Python 3.10
* GPU nvidia 
  * CUDA version 11.2
  * CUDNN verssion 8.1.0
* Conda
* Keras/Tensorflow

## Instruction d'installation
### Création de l'nvironnement conda
Pour créer un environnement avec Python 3.10 utilisant Conda, exécutez les commandes suivantes :
1. Création de l'environnement conda avec python 3.10
```
conda create -n py310 python=3.10
```

2. Activation de l'environnement conda
```
conda activate py310
```

### Installation de CUDA et cuDNN
3. Dans l'environnement Conda, installez les versions spécifiques de CUDA Toolkit et cuDNN nécessaires pour TensorFlow (GPU)
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

### Installation de kera/tensorflow pour gpu
4. Installez TensorFlow avec prise en charge GPU
```
python -m pip install "tensorflow<2.11"
```

## Installation des Dépendances
1. Cloner le répertoire 
```
git clone https://github.com/User-vin/txt-classification-camembert.git
cd <nom_du_dossier>
```

2. Installer les dépendances
```
pip install -r path/to/requirements.txt
```

## Structure du répertoire
```plaintext
project-name/
│
├── data/
│   ├── corpus/               # Données brutes
│   │   ├── train/            # Données traitées
│   │   ├── test/             # Données traitées
│   │   └── val/              # Données traitées
│   └── pkl/                  # Explications sur les données
│       ├── train.pkl         # Explications sur les données
│       ├── test.pkl          # Explications sur les données
│       ├── val.pkl           # Explications sur les données
│       ├── train_reduced.pkl # Explications sur les données
│       ├── test_reduced.pkl  # Explications sur les données
│       └── val_reduced.pkl   # Explications sur les données
│
├── results/
│   ├── figures/              # Graphiques et illustrations
│   ├── model_1/              # Journaux d'entraînement
│   └── model_2/              # Explications sur les résultats
│
├── scripts/
│   ├── config.py             # Tests pour le prétraitement des données
│   ├── data.py               # Tests pour le prétraitement des données
│   ├── main.py               # Tests pour le prétraitement des données
│   ├── model_1.py            # Tests pour le modèle 1
│   └── model_2.py            # Tests pour le modèle 2
│
├── README.md                 # Présentation du projet
└── requirements.txt          # Liste des dépendances
```

# Utilisation du Dépôt
## Données
Dans le dossier data 3 dossiers : train, test et val, déposé les textes romanesques francais au format suivants : 

Attention : Chaque texte est unique, et ne peut pas être présent dans les 3 dossiers 

Le seul prétraitement effectué, permet d'isoler toutes ponctuations du reste du texte, donc appliqué votre propre prétraiment si nécessaire.

```
(Nom auteur)(Prénom auteur)(Titre de l'oeuvre)(Sexe de l'auteur, 1:Homme ou 2:Femme)(Années de parution)()()()().txt
```
Ex : 
```
(ABEL)(Barbara)(l_innocence des bourreaux)(2)(2015)(1969)(v)(fr)(z)(z)(V)(z)(T).txt
```

Attention : Toutes les données formatées, tokenizé et ainsi que tous les labels necéssaire à l'entraînement d'un modèle sont compris dans des fichier .pkl non inclu avec le répertoire. 
Tous ces fichiers sont crées et enregistrés dans le dossier pkl.

Depuis l'invité de commande

1. Se positionner dans le dossier data
```
cd path/to/data
```

2. Créer le dossier pkl s'il n'est pas déjà présent
```
mkdir pkl
```

Les données nécessaire lors de l'entrainement et de l'évaluation sont chargées depuis les fichiers .pkl, s'ils n'existent pas, alors ils seront générés au moment de l'exécution du fichier data.py ou durant l'entrainement d'un modèle qui appel obligatoirement le fichier data.py.

Pour changer les données ou le taux de données utilisées durant l'entrainement et l'évaluation, il est nécessaire de supprimer les fichiers reduced .pkl utilisés 


## Configuration de quelques paramètres de l'entraînement 
Modifier le contenu des variables contenu dans le fichier config.py, tous les paramètres ne sont pas personnalisable depuis ce fichiers.

Certain d'entres eux nécessite une modification en dur directement dans le dossier .py correspondant au modèle.


## Entraînement & évaluation des Modèles 
Dans votre invité de commande
1. Activer votre environnement conda
```
conda activate nom_environement
```

2. Se positionner dans le dossier scripts
```
cd path/to/scripts
```

3. Entraînement, deux possibilités, 

    3.1. Entraîner un seul modèle
    Exécuter le script correspondant au modèle ex: mm_cnn.py
    ```
    python model.py
    ```
    3.2. Entrainé tous les modèles à la suite, exécuter le fichier main.py
    ```
    python main.py
    ```

L'entraînement se lance alors, et plus aucun input de l'utilisateur n'est requis

## Modifier un modèle existant
Ouvrir le fichier .py correspondant et modifier les paramètes en dur qui y sont présents.

# Méthodologie 
* Pré traitement des textes pour isoler toute ponctuation 
ex : "oui!" devient "oui !"
* Chargement d'un modèle de type Bert pré entrainé --> Distil Camembert https://huggingface.co/cmarkea/distilcamembert-base  


# Résultats
## Modèle par modèles
## Comparaison des modèles
## Métriques

# Conclusion
## 



