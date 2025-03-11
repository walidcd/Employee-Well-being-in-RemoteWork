from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from data_loader import load_data

# Spécifier le chemin du dataset
dataset_path = "./dataset"

# Redimensionner les images et effectuer une normalisation pour l'évaluation
test_datagen = ImageDataGenerator(rescale=1./255)

# Utilisation de ImageDataGenerator pour charger les données de test
test_generator = test_datagen.flow_from_directory(
    dataset_path + '/test',  # Assurez-vous de spécifier le bon chemin
    target_size=(64, 64),    # Redimensionner les images à 64x64
    batch_size=32,           # Augmenter la taille du batch pour plus d'efficacité
    class_mode='binary'
)

# Charger le modèle MobileNet amélioré
model = load_model("../models/posture_model_improved.keras")

# Évaluer le modèle sur les images de test
test_loss, test_acc = model.evaluate(test_generator)

# Afficher la précision de test
print(f"Test Accuracy: {test_acc:.2f}")