## MAIN PAN

import mm_cnn_model_training 
import ss_cnn_model_training

import tensorflow as tf
import time

def wait_for_cooldown(seconds):
    """Fait une pause pour un nombre de secondes donné.

    Args:
        seconds (int): Le nombre de secondes à attendre.
    """
    print(f"Pause de {seconds} secondes pour refroidir...")
    time.sleep(seconds)

if __name__ == '__main__':
    
    durations = {}
    start_time = time.time()
    
    
    # ----------------------------------------------------------------
    
    
    print("\n\n=================================================")
    print("=           Entraînement du modèle mm_cnn        =")
    print("=================================================\n\n")
    
    print("Entraînement du modèle mm_cnn en cours...\n")
    start_mm_cnn = time.time()
    mm_cnn_model_training.main()
    end_mm_cnn = time.time()
    tf.keras.backend.clear_session()  # Nettoyer la session TensorFlow

    durations["mm_cnn"] = end_mm_cnn - start_mm_cnn
    wait_for_cooldown(60)  # Attendre 60 secondes pour refroidir
    
    
    # ----------------------------------------------------------------
    
    
    print("\n\n=================================================")
    print("=           Entraînement du modèle ss_cnn        =")
    print("=================================================\n\n")

    print("Entraînement du modèle ss_cnn en cours...\n")
    start_ss_cnn = time.time()
    ss_cnn_model_training.main()
    end_ss_cnn = time.time()
    tf.keras.backend.clear_session()  # Nettoyer la session TensorFlow

    durations["ss_cnn"] = end_ss_cnn - start_ss_cnn


    # ----------------------------------------------------------------


print("\n=================================================")
print("=   Tous les modèles ont été entraînés avec succès  =")
print("=================================================\n")
print("La mémoire a été libérée.")

durations["total"] = time.time() - start_time

print(durations)
