# file: compare_models.py
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import librosa
import librosa.display
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# -----------------------------
# Paths (adjust if needed)
# -----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
AUDIO_PATH = os.path.join(PROJECT_ROOT, "dataset", "audio")
FEATURES_PATH = os.path.join(MODEL_DIR, "features.npy")
LABELS_PATH = os.path.join(MODEL_DIR, "labels.npy")
CNN_MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.keras")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.pkl")
TMP_SPEC_PATH = os.path.join(PROJECT_ROOT, "scripts", "tmp_spec.png")

ML_MODELS = {
    "Random Forest": os.path.join(MODEL_DIR, "rf_model.joblib"),
    "KNN": os.path.join(MODEL_DIR, "knn_model.joblib"),
    "SVM": os.path.join(MODEL_DIR, "svm_model.joblib")
}

# -----------------------------
# Load features & labels
# -----------------------------
X = np.load(FEATURES_PATH)
y = np.load(LABELS_PATH)
print(f"✅ Loaded features {X.shape} and labels {y.shape}")

# -----------------------------
# Evaluate ML models
# -----------------------------
ml_accuracies = {}
for name, path in ML_MODELS.items():
    model = joblib.load(path)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    ml_accuracies[name] = acc
    print(f"{name} Accuracy: {acc:.3f}")

# -----------------------------
# CNN: audio → spectrogram (skip unreadable files)
# -----------------------------
def audio_to_spectrogram_image(audio_file, duration=30, img_size=(224,224)):
    try:
        y, sr = librosa.load(audio_file, duration=duration)
    except Exception as e:
        print(f"⚠️ Skipping {audio_file}: {e}")
        return None  # Skip this file

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3,3))
    plt.axis('off')
    librosa.display.specshow(S_db, sr=sr, cmap='viridis')
    plt.savefig(TMP_SPEC_PATH, bbox_inches='tight', pad_inches=0)
    plt.close()

    img = load_img(TMP_SPEC_PATH, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    os.remove(TMP_SPEC_PATH)
    return img_array

# -----------------------------
# Evaluate CNN on audio dataset
# -----------------------------
cnn_model = load_model(CNN_MODEL_PATH)
index_to_genre = {v:k for k,v in joblib.load(CLASS_INDICES_PATH).items()}

y_true = []
y_pred = []

for genre in os.listdir(AUDIO_PATH):
    genre_path = os.path.join(AUDIO_PATH, genre)
    if not os.path.isdir(genre_path):
        continue
    for file in os.listdir(genre_path):
        if not file.lower().endswith((".wav", ".mp3")):
            continue
        audio_file = os.path.join(genre_path, file)
        img_array = audio_to_spectrogram_image(audio_file)
        if img_array is None:  # Skip unreadable files
            continue
        pred_index = np.argmax(cnn_model.predict(img_array), axis=1)[0]
        pred_genre = index_to_genre[pred_index]

        y_true.append(genre)
        y_pred.append(pred_genre)

cnn_acc = accuracy_score(y_true, y_pred)
print(f"CNN Accuracy: {cnn_acc:.3f}")

cm = confusion_matrix(y_true, y_pred, labels=list(index_to_genre.values()))
disp = ConfusionMatrixDisplay(cm, display_labels=list(index_to_genre.values()))
disp.plot(xticks_rotation=45)
plt.title("CNN Confusion Matrix")
plt.show()

# -----------------------------
# Print all accuracies together
# -----------------------------
print("\n=== All Model Accuracies ===")
for name, acc in ml_accuracies.items():
    print(f"{name}: {acc:.3f}")
print(f"CNN: {cnn_acc:.3f}")
