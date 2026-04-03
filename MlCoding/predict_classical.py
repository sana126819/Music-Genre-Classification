import os
import librosa
import numpy as np
import joblib

def extract_features(audio_file, duration=30, n_mfcc=40):
    y, sr = librosa.load(audio_file, duration=duration)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean.reshape(1, -1)

class ClassicalClassifier:
    def __init__(self):
        self.models = {
            "RandomForest": joblib.load("../models/rf_model.pkl"),
            "SVM": joblib.load("../models/svm_model.pkl"),
            "KNN": joblib.load("../models/knn_model.pkl")
        }
        self.class_indices = joblib.load("../models/class_indices.pkl")
        self.index_to_genre = {v:k for k,v in self.class_indices.items()}

    def predict(self, audio_file, algo="RandomForest"):
        if algo not in self.models:
            raise ValueError(f"Algorithm {algo} not supported")
        features = extract_features(audio_file)
        pred_index = self.models[algo].predict(features)[0]
        return self.index_to_genre[pred_index]
