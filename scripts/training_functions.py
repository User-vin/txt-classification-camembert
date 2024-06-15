
# Import 
import os
import config
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from transformers import TFCamembertModel
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    f1_score, 
    precision_score, 
    recall_score, 
    roc_curve,
    roc_auc_score 
)


def calculate_estimated_year_tensor(intervals, probabilities):
    def get_bounds(interval: str):
        start, end = interval.strip("[]()").split(", ")
        return int(start), int(end)
    
    values = []
    for i, interval in enumerate(intervals):
        start, end = get_bounds(interval)
        if i == 0:
            values.append(start)
        elif i == len(intervals) - 1:
            values.append(end)
        else:
            values.append((start + end) / 2)
    
    values_tensor = tf.constant(values, dtype=tf.float32)
    estimated_date = tf.reduce_sum(values_tensor * probabilities, axis=1)
    estimated_year = tf.round(estimated_date)
    
    return estimated_year


def custom_loss(y_true, y_pred):
    # Obtenez les années estimées en utilisant les prédictions fournies par le modèle
    # intervals = ['[1825, 1850)', '[1850, 1875)', '[1875, 1900)', '[1900, 1925)', '[1925, 1950)', '[1950, 1975)', '[1975, 2000)', '[2000, 2024)']
    estimated_years = calculate_estimated_year_tensor(list(config.DATE_MAP.values()), y_pred)
    # estimated_years = calculate_estimated_year_tensor(intervals, y_pred)
    # Convertir y_true en float32 sans changer la shape
    y_true_float = tf.cast(y_true, tf.float32)
    # Reshape y_true_float : passer de [16 1] à [16]
    y_true_float = tf.reshape(y_true_float, [-1])
    # Calculez les erreurs carrées entre les années estimées et les valeurs réelles
    squared_errors = tf.square(estimated_years - y_true_float)
    # Sommez les carrés des erreurs pour obtenir la somme totale des erreurs
    sum_of_squared_errors = tf.reduce_sum(squared_errors)
    # Calculer le nombre total d'échantillons
    num_samples = tf.cast(tf.shape(y_true_float)[0], tf.float32)
    # Calculer la moyenne des erreurs carrées
    mean_squared_error = sum_of_squared_errors / num_samples
    
    return mean_squared_error


def custom_objects_dict():
    cutom_objects = {
        "TFCamembertModel": TFCamembertModel, 
        "custom_loss": custom_loss, 
        "custom_metric": custom_metric,
    }
    return cutom_objects


def custom_metric(y_true, y_pred):
    # intervals = ['[1825, 1850)', '[1850, 1875)', '[1875, 1900)', '[1900, 1925)', '[1925, 1950)', '[1950, 1975)', '[1975, 2000)', '[2000, 2024)']
    estimated_years = calculate_estimated_year_tensor(list(config.DATE_MAP.values()), y_pred)
    # estimated_years = calculate_estimated_year_tensor(intervals, y_pred)
    # Convertir y_true en float32 sans changer la shape
    y_true_float = tf.cast(y_true, tf.float32)
    # Aplatir y_true pour correspondre à estimated_years
    y_true_flat = tf.reshape(y_true_float, [-1])
    # Calcul de la précision
    correct_predictions = tf.abs(estimated_years - y_true_flat) <= 25
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy


def evaluate_model(df: pd.DataFrame, confusion_matrix_output: str, roc_curve_output: str, json_output: str):
    plt.style.use("seaborn-v0_8-whitegrid")
    true_sexe = np.array(df["true sexe"].tolist())
    pred_sexe = np.array(df["pred sexe"].tolist())
    true_date = np.array(df["true date"].tolist())

    # Sexe metrics
    accuracy_sexe = accuracy_score(true_sexe, pred_sexe)
    precision_sexe = precision_score(true_sexe, pred_sexe)
    recall_sexe = recall_score(true_sexe, pred_sexe)
    f1_sexe = f1_score(true_sexe, pred_sexe)
    auc_sexe = roc_auc_score(true_sexe, pred_sexe)

    print(f"\nSexe - Accuracy: {accuracy_sexe} \nPrecision: {precision_sexe} \nRecall: {recall_sexe} \nF1 Score: {f1_sexe} \nAUC: {auc_sexe}\n")

    # Enregistrement des métriques dans un .json
    metrics = {
        "accuracy_sexe": accuracy_sexe,
        "precision_sexe": precision_sexe,
        "recall_sexe": recall_sexe,
        "f1_sexe": f1_sexe,
        "auc_sexe": auc_sexe
    }

    with open(json_output, "w") as json_file:
        json.dump(metrics, json_file, indent=4)

    # Confusion matrix

    plt.style.use("seaborn-v0_8-white")
    # Convert predictions to binary
    pred_binary = (pred_sexe > 0.5).astype(int)
    # Calculate confusion matrix
    cm = confusion_matrix(true_sexe, pred_binary)
    # Normalize the confusion matrix to display rates
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # Plot confusion matrix with updated display labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=["Femme", "Homme"])
    disp.plot(cmap=plt.cm.Blues, values_format=".2f")  # Set the colormap to Blues and format to 2 decimal places
    plt.title(f"Matrice de confusion - Sexe")
    plt.savefig(confusion_matrix_output)
    plt.close()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(true_sexe, pred_sexe)
    plt.figure()
    plt.plot(fpr, tpr, color="blue", lw=2, label="ROC curve (AUC = %0.2f)" % auc_sexe)
    plt.plot([0, 1], [0, 1], color="grey", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic - Sexe")
    plt.legend(loc="lower right")
    plt.savefig(roc_curve_output)
    plt.close()

    return accuracy_sexe, precision_sexe, recall_sexe, f1_sexe, auc_sexe


def map_true_date_to_interval(date):
    for _, value in config.DATE_MAP.items():
        start, end = map(int, value.strip("[]()").split(", "))
        if start <= date < end:
            return value
    return None  # Return None if date does not fall into any defined interval


def prediction(model, input_data, sexe_label, date_label=None):
    test_predictions = model.predict(input_data)

    if len(test_predictions) == 2:
        # Unpack predictions
        pred_sexe = np.squeeze((test_predictions[0] > 0.5).astype("int32"))
        pred_date = test_predictions[1]
    else:
        pred_sexe = np.squeeze((test_predictions > 0.5).astype("int32"))
        
    true_sexe = np.array(sexe_label).astype("int32")
    true_date = np.array(date_label).astype("int32") if date_label is not None else None
    
    df = pd.DataFrame({
        "true sexe": true_sexe,
        "pred sexe": pred_sexe,
        "true date": true_date,
        # "pred date": pred_date,
    })
    return df


def save_accuracy_by_interval_and_gender(df: pd.DataFrame, output_file: str):
    """
    Prépare les données pour calculer l'accuracy pour chaque sexe et chaque intervalle,
    puis trace un histogramme à barres pour l'accuracy par intervalle et sexe.

    Arguments :
    df : DataFrame - Le DataFrame contenant les données à analyser.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    # Préparer les données pour l'accuracy
    intervals = df["interval"].unique()
    data_for_plot = []

    for interval in intervals:
        df_interval = df[df["interval"] == interval]
        for sex in [0, 1]:  # 0: Femme, 1: Homme
            df_sex = df_interval[df_interval["true sexe"] == sex]
            cm = confusion_matrix(df_sex["true sexe"], df_sex["pred sexe"])
            total = cm.sum()
            correct_predictions = cm.trace()  # Sum of True Positives and True Negatives
            accuracy = correct_predictions / total
            # gender = "Femme" if sex == 0 else "Homme"
            gender = "Femme" if sex == 0 else "Homme"
            data_for_plot.append((interval, gender, accuracy))
    
    df_accuracy = pd.DataFrame(data_for_plot, columns=["Interval", "Gender", "Accuracy"])

    # Tracer l'histogramme à barres pour l'accuracy par intervalle et sexe
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x="Interval", y="Accuracy", hue="Gender", data=df_accuracy, palette=["#FF7F0E", "#1F77B4"])
    plt.xlabel("Intervalles")
    plt.ylabel("Taux d\'accuracy")
    plt.title("Taux d\'accuracy par intervalle et sexe")
    plt.xticks(rotation=45)
    # plt.legend(title="Sexe")
    plt.legend(title="Sexe", loc="upper right", frameon=True)
    # plt.legend(title="Sexe", loc='upper right', fontsize=12, title_fontsize=14, frameon=True, fancybox=False, framealpha=1, shadow=False, borderpad=1)
    plt.ylim(0, 1)  # Limiter l'axe Y entre 0 et 1

    # Ajouter les valeurs d'accuracy au-dessus de chaque barre
    for p in ax.patches:
        if p.get_height() != 0.00:
            ax.annotate(f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha="center", va="bottom", fontsize=8, color="black", xytext=(0, 5), textcoords="offset points")

    plt.savefig(output_file)
    plt.close()


def save_hist_confusion_matrix(df: pd.DataFrame, output_file: str):
    """
    Prépare les données et trace l'histogramme de la matrice de confusion normalisée par intervalle et sexe.

    Arguments :
    df : DataFrame - Le DataFrame contenant les données à analyser.

    Renvoie :
    None
    """
    def prepare_confusion_data(df: pd.DataFrame):
        """
        Prépare les données pour la matrice de confusion normalisée.

        Arguments :
        df : DataFrame - Le DataFrame contenant les données à analyser.

        Renvoie :
        DataFrame - Un DataFrame contenant les données de la matrice de confusion normalisée.
        """
        # Obtenir les intervalles uniques
        intervals = df["interval"].unique()
        data_for_plot = []

        # Parcourir chaque intervalle
        for interval in intervals:
            # Filtrer le DataFrame pour l'intervalle actuel
            df_interval = df[df["interval"] == interval]
            # Parcourir les deux sexes (0: Femme, 1: Homme)
            for sex in [0, 1]:
                # Filtrer le DataFrame pour le sexe actuel
                df_sex = df_interval[df_interval["true sexe"] == sex]
                # Calculer la matrice de confusion
                cm = confusion_matrix(df_sex["true sexe"], df_sex["pred sexe"])
                # Calculer le total des prédictions
                total = cm.sum()
                # Normaliser la matrice de confusion
                cm_normalized = cm / total
                # Déterminer le genre (Femme ou Homme)
                gender = "Femme" if sex == 0 else "Homme"
                # Parcourir la matrice de confusion normalisée
                for i, row in enumerate(cm_normalized):
                    for j, val in enumerate(row):
                        # Déterminer la catégorie (Vrai ou Faux)
                        category = "True" if i == j else "False"
                        data_for_plot.append((interval, f"{category} {gender}", val))
        
        # Créer un DataFrame à partir des données
        return pd.DataFrame(data_for_plot, columns=["Interval", "Category", "Value"])
    plt.style.use("seaborn-v0_8-whitegrid")
    # Préparer les données
    df_plot = prepare_confusion_data(df)

    # Utiliser la palette tab10 de seaborn
    colors = sns.color_palette("tab10")

    # Créer une palette avec les couleurs spécifiées pour chaque catégorie
    custom_palette = {"True Femme": colors[1], "True Homme": colors[0]}
    palette = {**custom_palette, **{category: colors[i+2] for i, category in enumerate(df_plot["Category"].unique()) if category not in custom_palette}}

    # Tracer l'histogramme à barres avec la palette personnalisée
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x="Interval", y="Value", hue="Category", data=df_plot, palette=palette)
    plt.xlabel("Intervalles")
    plt.ylabel("Proportion")
    plt.title("Histogramme des matrices de confusions par intervalle et sexe")
    plt.xticks(rotation=45)
    # plt.legend(title="Prédictions")
    plt.legend(title="Prédictions", loc="upper right", frameon=True)
    plt.ylim(0, 1)  # Limiter l'axe Y entre 0 et 1

    # Ajouter les valeurs normalisées au-dessus de chaque barre
    for p in ax.patches:
        if p.get_height() != 0.00:
            ax.annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha="center", va="bottom", fontsize=8, color="black", xytext=(3, 5), textcoords="offset points")

    plt.savefig(output_file)
    plt.close()


def save_training_history(history_data: dict, save_dir: str="plots", base_filename: str="plot", single_figure: bool=False):
    """
    Save the training history for a model and save the figures.

    Parameters:
    history (History): The training history returned by model.fit()
    save_dir (str): Directory to save the plots
    base_filename (str): Base filename for the saved plots
    single_figure (bool): Whether to save all plots in a single figure
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Function to annotate values on the plot
    def annotate_values(ax, x, y):
        for i, txt in enumerate(y):
            ax.annotate(round(txt, 2), (x[i], y[i]), textcoords="offset points", xytext=(0, 5), ha="center")

    # Set the style
    plt.style.use("seaborn-v0_8-whitegrid")

    if single_figure:
        # Determine the number of unique keys (excluding validation keys)
        unique_keys = set(key.replace("val_", "") for key in history_data.keys())
        num_plots = len(unique_keys)
        fig, axs = plt.subplots(num_plots, 1, figsize=(15, num_plots * 5))
        
        if num_plots == 1:
            axs = [axs]

        for i, key in enumerate(unique_keys):
            train_key = key
            val_key = "val_" + key

            axs[i].plot(history_data[train_key], label=f"Training {train_key.replace('_', ' ').title()}", marker="o")
            if val_key in history_data:
                axs[i].plot(history_data[val_key], label=f"{val_key.replace('_', ' ').title()}", marker="o")
            
            axs[i].set_xlabel("Epochs")
            axs[i].set_ylabel(train_key.split("_")[-1].title())
            axs[i].set_title(train_key.replace("_", " ").title())
            axs[i].legend()
            axs[i].grid(True)

            annotate_values(axs[i], range(len(history_data[train_key])), history_data[train_key])
            if val_key in history_data:
                annotate_values(axs[i], range(len(history_data[val_key])), history_data[val_key])
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{base_filename}_combined.png"))
        plt.close()
    else:
        for key in set(key.replace("val_", "") for key in history_data.keys()):
            train_key = key
            val_key = "val_" + key

            plt.figure(figsize=(8, 6))
            plt.plot(history_data[train_key], label=f"Train {train_key.replace('_', ' ').title()}", marker="o")
            if val_key in history_data:
                plt.plot(history_data[val_key], label=f"{val_key.replace('_', ' ').title()}", marker="o")

            plt.xlabel("Epochs")
            plt.ylabel(train_key.split("_")[-1].title())
            plt.title(train_key.replace("_", " ").title())
            plt.legend()
            plt.grid(True)
            
            annotate_values(plt.gca(), range(len(history_data[train_key])), history_data[train_key])
            if val_key in history_data:
                annotate_values(plt.gca(), range(len(history_data[val_key])), history_data[val_key])
            
            plt.savefig(os.path.join(save_dir, f"{base_filename}_{train_key}.png"))
            plt.close()

