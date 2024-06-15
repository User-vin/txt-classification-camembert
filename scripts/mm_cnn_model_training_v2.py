import config
import data
import os

from transformers import TFAutoModel
import tensorflow as tf

from keras.models import Model
from keras.utils.vis_utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import (
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    # Embedding,
    Flatten,
    Input,
    MaxPooling1D,
)

import json
import numpy as np
import pprint


from training_functions import (
    # calculate_estimated_year_tensor,
    custom_loss,
    # custom_objects_dict,
    custom_metric,
    evaluate_model,
    map_true_date_to_interval,
    prediction,
    save_accuracy_by_interval_and_gender,
    save_hist_confusion_matrix,
    save_training_history,
)



def mm_cnn_v2(
    num_date_classes:int ,
    model_id: str=None,
    max_length: int=514,
    dense_units: int=16,
    conv_filters: int=32, 
    conv_kernel_size: int=9, 
):
    """
    Crée un modèle Keras avec un modèle BERT pré-entraîné pour une tâche multitâche de classification.

    Parameters:
    - model_id: str, identifiant du modèle pré-entraîné à utiliser (par exemple, 'bert-base-uncased')
    - max_length: int, la longueur maxiHomme des séquences d'entrée
    - dense_units: int, nombre d'unités pour les couches denses individuelles
    - concat_dense_units: int, nombre d'unités pour la couche dense après concaténation

    Returns:
    - model: Keras Model, le modèle compilé
    """
    # Charger le modèle BERT pré-entraîné
    bert_model = TFAutoModel.from_pretrained(model_id)

    # Définir les entrées du modèle
    input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")

    # Passer les entrées dans le modèle BERT
    bert_output = bert_model(input_ids, attention_mask=attention_mask) 
    sequence_output = bert_output.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)  Utiliser ensuite avec un CNN
    # pooled_output = bert_output.pooler_output  # Shape: (batch_size, hidden_size) # Utiliser le pooled_output pour la classification

    # CNN
    conv_layer = Conv1D(filters=conv_filters, kernel_size=conv_kernel_size, activation="relu", name="Conv1D")(sequence_output)
    pooling_layer = MaxPooling1D(pool_size=2, name="MaxPooling1D")(conv_layer)
    flatten_layer = Flatten(name="Flatten")(pooling_layer)
    dropout_layer = Dropout(0.3, name="Dropout")(flatten_layer) # Ajout des couches supplémentaires pour les tâches spécifiques

    # Dense layers for individual tasks
    dense_layer = Dense(units=dense_units, activation="relu", name="Dense")(dropout_layer)

    # Output layers for individual tasks
    sexe_output = Dense(1, activation="sigmoid", name="Sexe_output")(dense_layer)
    date_output = Dense(num_date_classes, activation="softmax", name="Date_output")(dense_layer)

    # Créer le modèle
    model = Model(inputs=[input_ids, attention_mask], outputs=[sexe_output, date_output])

    return model


def main():
    # 1 : Charger les données d'entraînements et de validation
    inputs_and_labels = data.main()

    train_inputs = inputs_and_labels["train_inputs"]
    train_sexe_labels = inputs_and_labels["train_sexe_labels"]
    train_date_labels = inputs_and_labels["train_date_labels"]

    val_inputs = inputs_and_labels["val_inputs"]
    val_sexe_labels = inputs_and_labels["val_sexe_labels"]
    val_date_labels = inputs_and_labels["val_date_labels"]

    # Libérer la mémoire occupée par inputs_and_labels
    del inputs_and_labels

    # Créer le répertoire s'il n'existe pas
    os.makedirs(config.MM_CNN_V2_RESULT_PATH, exist_ok=True)

    # 1. Intialiser le modèle
    model = mm_cnn_v2(num_date_classes=len(config.DATE_MAP), model_id=config.MODEL_ID)

    # 2. Compiler le modèle avec les fonctions de perte appropriées pour chaque sortie
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss={
            "Sexe_output": "binary_crossentropy", 
            "Date_output": custom_loss,
        },
        loss_weights={
            "Sexe_output": config.SEXE_LOSS_WEIGHT,
            "Date_output": config.DATE_LOSS_WEIGHT
        },
        metrics={
            "Sexe_output": "accuracy", 
            "Date_output": custom_metric,
        }
    )

    # Sauvegarder l'architecture du modèle en .png
    plot_model(model=model, show_shapes=True, to_file=config.MM_CNN_V2_RESULT_PATH + f"{config.MM_CNN_V2}_model_arch.png")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        filepath=config.MM_CNN_V2_RESULT_PATH + f"{config.MM_CNN_V2}_best_model.h5",
        monitor="val_loss",
        save_best_only=True,
        mode="min", # Sauvegarder le modèle avec la perte minimale
        verbose=1
    )

    # Early stopping
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=config.PATIENCE,
        verbose=1, # Affichage d'un message
        restore_best_weights=True # Restaurer les poids du meilleur modèle après l'arrêt
    )

    # 3. Entraîner le modèle
    print("\nDébut entraînement mm_cnn_v2\n")
    history = model.fit(
        x=train_inputs,
        y={
            "Sexe_output": train_sexe_labels, 
            "Date_output": train_date_labels,
            },
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[
            checkpoint_callback, 
            early_stopping_callback,
        ],
        validation_data=(
            val_inputs, 
            {
                "Sexe_output": val_sexe_labels, 
                "Date_output": val_date_labels,
            }),
    )
    print("\Fin entraînement mm_cnn_v2\n")
        
    # Sauvegarder l'historique de l'entraînement en .json
    with open(config.MM_CNN_V2_RESULT_PATH + f"{config.MM_CNN_V2}_history.json", "w") as json_file:
        json.dump(history.history, json_file, indent=4)
    # Sauvegarder l'historique de l'entraînement en .png
    save_training_history(history_data=history.history, save_dir=config.MM_CNN_V2_RESULT_PATH, base_filename=config.MM_CNN_V2, single_figure=True)
    save_training_history(history_data=history.history, save_dir=config.MM_CNN_V2_RESULT_PATH, base_filename=config.MM_CNN_V2, single_figure=False)


    # Evaluation --------------------------------


    # 1. Charger les données de tests
    inputs_and_labels = data.main()

    test_inputs = inputs_and_labels["test_inputs"]
    test_sexe_labels = inputs_and_labels["test_sexe_labels"]
    test_date_labels = inputs_and_labels["test_date_labels"]

    # Libérer la mémoire occupée par inputs_and_labels
    del inputs_and_labels

    # Évaluer le modèle sur les données de test avec le GPU
    evaluation_results = model.evaluate(
        x = test_inputs,
        y = {
            "Sexe_output": np.array(test_sexe_labels),
            "Date_output": np.array(test_date_labels),
        },
        return_dict=True
    )
        
    print("\nEvaluation...\n")
    pprint.pp(evaluation_results)

    # Sauvegarder les résultats de l'évaluation
    with open(config.MM_CNN_V2_RESULT_PATH + f"{config.MM_CNN_V2}_evaluation_results.json", "w") as json_file:
        json.dump(evaluation_results, json_file, indent=4)
        
    # Charger le modèle pré entraîné
    # model = load_model(
    #     filepath=config.MM_CNN_V2_RESULT_PATH + f"{config.MM_CNN_V2}_best_model.h5", 
    #     custom_objects=custom_objects_dict()
    # )

    print("\nPrédictions...\n")
    prediction_df = prediction(model=model, input_data=test_inputs, sexe_label=test_sexe_labels, date_label=test_date_labels)

    evaluate_model(
        df=prediction_df, 
        confusion_matrix_output=config.MM_CNN_V2_RESULT_PATH + f"{config.MM_CNN_V2}_confusion_matrix.png", 
        roc_curve_output=config.MM_CNN_V2_RESULT_PATH + f"{config.MM_CNN_V2}_roc_curve.png",
        json_output=config.MM_CNN_V2_RESULT_PATH + f"{config.MM_CNN_V2}_predictions_metrics.json"
    )

    # prediction_df["interval"] = prediction_df["true date"].map(config.DATE_MAP)
    prediction_df["interval"] = prediction_df["true date"].apply(map_true_date_to_interval)

    save_hist_confusion_matrix(df=prediction_df, output_file=config.MM_CNN_V2_RESULT_PATH + f"{config.MM_CNN_V2}_hist_confusion_matrix.png")

    save_accuracy_by_interval_and_gender(df=prediction_df, output_file=config.MM_CNN_V2_RESULT_PATH + f"{config.MM_CNN_V2}_accuracy_by_interval_sexes.png")


if __name__ == "__main__":
    main()




