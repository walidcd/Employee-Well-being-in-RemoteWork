# train.py
from data_loader import load_data
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Sequential


dataset_path = "./dataset"

# Augmentation des données
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# Chargement des données avec augmentation
train_generator = train_datagen.flow_from_directory(
    dataset_path + '/train',  
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

valid_generator = valid_datagen.flow_from_directory(
    dataset_path + '/valid',  
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_generator = valid_datagen.flow_from_directory(
    dataset_path + '/test', 
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Création du modèle avec MobileNet pré-entraîné
def create_model():
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
    base_model.trainable = False  # Congeler les poids du modèle pré-entraîné

    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout pour éviter le sur-apprentissage
    model.add(Dense(1, activation='sigmoid'))

    lr_schedule = ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True
    )

    model.compile(optimizer=SGD(learning_rate=lr_schedule, momentum=0.9), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    return model

# Création du modèle
model = create_model()

# Callback pour réduire le learning rate si stagnation
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # Diviser le learning rate par 2
    patience=3,        # Après 3 epochs sans amélioration
    min_lr=1e-6,
    verbose=1
)

# EarlyStopping pour stopper l'entraînement si la performance de validation ne s'améliore plus
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Définition des epochs - plus d'époques pour permettre l'apprentissage
EPOCHS = 20

# Entraînement du modèle
history = model.fit(
    train_generator, 
    validation_data=valid_generator, 
    epochs=EPOCHS,
    callbacks=[reduce_lr, early_stop]  # Ajouter les callbacks
)

# Sauvegarde du modèle après l'entraînement
model.save("../models/posture_model_improved.keras")

# Fonction pour tracer la précision et la perte
def plot_training(history):
    acc, val_acc = history.history['accuracy'], history.history['val_accuracy']
    loss, val_loss = history.history['loss'], history.history['val_loss']
    epochs_range = range(len(acc))
    
    # Configuration des sous-graphes
    plt.figure(figsize=(12, 4))
    
    # Tracer la précision
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend()
    plt.title("Training vs Validation Accuracy")
    
    # Tracer la perte
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend()
    plt.title("Training vs Validation Loss")
    
    plt.savefig("training_plot_improved.png")
    print("Graphique sauvegardé dans 'training_plot_improved.png'.")

# Appeler la fonction pour afficher les courbes de précision et perte
plot_training(history)

# Évaluer sur l'ensemble de test
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")