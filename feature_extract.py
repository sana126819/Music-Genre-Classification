
 # file: music_feature_extractor.py
import os
import librosa
import numpy as np

class MusicFeatureExtractor:
    def __init__(self, dataset_path="./dataset/audio", n_mfcc=13, duration=30, save_path="./models"):
        self.dataset_path = dataset_path
        self.n_mfcc = n_mfcc
        self.duration = duration
        self.save_path = save_path
        self.features = []
        self.labels = []

    def process_file(self, file_path, genre):
        try:
            y, sr = librosa.load(file_path, duration=self.duration)

            if len(y) < sr * 1.0:
                print(f"Skipped {file_path}: too short")
                return

            # Normalize audio
            y = y / np.max(np.abs(y))

            # Extract MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc.T, axis=0)

            if np.isnan(mfcc_mean).any() or np.isinf(mfcc_mean).any():
                print(f"Skipped {file_path}: invalid MFCC")
                return

            self.features.append(mfcc_mean)
            self.labels.append(genre)

        except Exception as e:
            print(f"Skipped {file_path}: {e}")

    def process_dataset(self):
        for genre in os.listdir(self.dataset_path):
            genre_path = os.path.join(self.dataset_path, genre)
            if not os.path.isdir(genre_path):
                continue
            print(f"Processing genre: {genre}")
            for fname in os.listdir(genre_path):
                file_path = os.path.join(genre_path, fname)
                if not file_path.lower().endswith((".wav", ".mp3")):
                    continue
                self.process_file(file_path, genre)

    def save_features(self):
        os.makedirs(self.save_path, exist_ok=True)
        X = np.array(self.features)
        y = np.array(self.labels)
        np.save(os.path.join(self.save_path, "features.npy"), X)
        np.save(os.path.join(self.save_path, "labels.npy"), y)
        print("✅ Features saved successfully!")
        print("Feature matrix shape:", X.shape)
        print("Number of labels:", len(y))

if __name__ == "__main__":
    extractor = MusicFeatureExtractor(
        dataset_path="../dataset/audio",  # adjust path if needed
        n_mfcc=13,
        duration=30,
        save_path="../models"
    )
    extractor.process_dataset()
    extractor.save_features()
