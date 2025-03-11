from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

IMG_SIZE = (64, 64)

model = load_model("../models/posture_model_improved.keras")

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return "Good Posture" if prediction[0][0] > 0.5 else "Bad Posture"

img_path = "C:\\Users\\salma\\Downloads\\Posture_Classification\posture_dataset\\test\\bad\\WIN_20250211_13_14_49_Pro_mp4-0002_jpg.rf.ca156f330c8ff00fedd81f66b7dafc4c.jpg"
print(f"Prediction: {predict_image(img_path)}")