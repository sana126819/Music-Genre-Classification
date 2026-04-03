# scripts/predict_genre.py
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import joblib

class GenreClassifier:
    def __init__(self, model_path, class_indices_path, img_size=(224,224), tmp_dir=None):
        # Load CNN model
        self.model = load_model(model_path)
        # Load class indices
        self.class_indices = joblib.load(class_indices_path)
        self.index_to_genre = {v:k for k,v in self.class_indices.items()}
        self.img_size = img_size
        self.tmp_dir = tmp_dir or os.getcwd()
        self.tmp_spec_path = os.path.join(self.tmp_dir, "tmp_spec.png")

    def predict_audio(self, audio_file, duration=30):
        """Predict genre from audio file."""
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        # Audio -> Mel spectrogram
        y, sr = librosa.load(audio_file, duration=duration)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)

        # Save spectrogram temporarily
        plt.figure(figsize=(3,3))
        plt.axis('off')
        librosa.display.specshow(S_db, sr=sr, cmap='viridis')
        plt.savefig(self.tmp_spec_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        # Load image for CNN prediction
        img = image.load_img(self.tmp_spec_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict genre
        pred_index = np.argmax(self.model.predict(img_array), axis=1)[0]
        genre = self.index_to_genre[pred_index]

        # Delete temporary image
        os.remove(self.tmp_spec_path)
        return genre
