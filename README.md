



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
conda create -n env_name python=3.10
```

2. Activation de l'environnement conda
```
conda activate env_name
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
Dans le dossier ``data``, trois sous-dossiers sont présents : ``train``, ``test`` et ``val``. Déposer les textes au format suivant :

```
(Nom auteur)(Prénom auteur)(Titre de l'oeuvre)(Sexe de l'auteur, 1:Homme ou 2:Femme)(Années de parution)()()()().txt
```
Ex : 
```
(ABEL)(Barbara)(l_innocence des bourreaux)(2)(2015)(1969)(v)(fr)(z)(z)(V)(z)(T).txt
```

Le seul prétraitement effectué isole toutes les ponctuations du reste du texte. Appliquez votre propre prétraitement si nécessaire.

``Attention : ``
* Chaque texte doit être unique et ne peut pas se trouver dans plusieurs dossiers.
* Toutes les données formatées, tokenisées et les labels nécessaires à l'entraînement du modèle sont contenus dans des fichiers .pkl non inclus dans le répertoire. Ces fichiers sont créés et enregistrés dans le dossier pkl en exécutant directement data.py ou lors de l'entraînement d'un modèle.

Les données nécessaires à l'entraînement et à l'évaluation sont chargées depuis les fichiers .pkl. S'ils n'existent pas, ils seront générés au moment de l'exécution de data.py ou durant l'entraînement d'un modèle qui appelle obligatoirement data.py.

Pour changer les données ou le taux de données utilisées durant l'entraînement et l'évaluation, il est nécessaire de supprimer les fichiers ``_reduced.pkl`` ou ``_balanced.pkl`` utilisés, puis modifier la quantité de données en ajustant les tailles dans le fichier config.py.

## Configuration de quelques paramètres de l'entraînement 
Modifiez le contenu des variables dans le fichier config.py. Tous les paramètres ne sont pas personnalisables depuis ce fichier. 

Certains nécessitent une modification directe dans le fichier .py correspondant au modèle.

## Entraînement & évaluation des Modèles 
Dans le terminal : 
1. Activer l'environnement conda
```
conda activate py310
```

2. Se positionner dans le dossier ``scripts``
```
cd path/to/scripts
```

3. Entraînement, deux possibilités :

    3.1. Entraîner un seul modèle
    Exécuter le script correspondant au modèle, par exemple ``mm_cnn``
    ```
    python mm_cnn.py
    ```
    3.2. Entrainé tous les modèles à la suite, exécuter le fichier main.py
    ```
    python main.py
    ```

L'entraînement se lance alors, et plus aucun input de l'utilisateur n'est requis.

## Modifier un modèle existant
Ouvrir le fichier ``.py`` correspondant et modifier les paramètes en dur présents dans ce fichier.

# BERT : CamemBERT


# CNN


# Multitâche et multitête


# Méthodologie 

* Pré traitement des textes pour isoler toute ponctuation 
Ex : ``"oui!"`` devient ``"oui !"``
* Chargement d'un pré entrîné modèle de type ``BERT`` pré entrainé : ``DistilCamembert`` https://huggingface.co/cmarkea/distilcamembert-base  


# Résultats
## Modèle par modèles
## Comparaison des modèles
## Métriques

# Conclusion
## 



