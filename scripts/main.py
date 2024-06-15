

import mm_cnn_model_training 
import mm_cnn_model_training_v2 
import ss_cnn_model_training

import tensorflow as tf


if __name__ == '__main__':
    print("\n\n=================================================")
    print("=           Entraînement du modèle mm_cnn        =")
    print("=================================================\n\n")

    # print("Entraînement du modèle mm_cnn en cours...\n")
    mm_cnn_model_training.main()
    tf.keras.backend.clear_session()  # Nettoyer la session TensorFlow

    print("\n\n=================================================")
    print("=         Entraînement du modèle mm_cnn_v2      =")
    print("=================================================\n\n")

    # print("Entraînement du modèle mm_cnn_v2 en cours...\n")
    mm_cnn_model_training_v2.main()
    tf.keras.backend.clear_session()  # Nettoyer la session TensorFlow

    print("\n\n=================================================")
    print("=           Entraînement du modèle ss_cnn        =")
    print("=================================================\n\n")

    # print("Entraînement du modèle ss_cnn en cours...\n")
    ss_cnn_model_training.main()
    tf.keras.backend.clear_session()  # Nettoyer la session TensorFlow

print("\n=================================================")
print("=   Tous les modèles ont été entraînés avec succès  =")
print("=================================================\n")
print("La mémoire a été libérée.")

