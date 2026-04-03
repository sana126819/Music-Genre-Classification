# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# %%
# Project root (assume this file is in scripts/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "image")
AUDIO_PATH = os.path.join(PROJECT_ROOT, "dataset", "audio")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "cnn_model.keras")
CLASS_INDICES_PATH = os.path.join(MODEL_DIR, "class_indices.pkl")
TMP_SPEC_PATH = os.path.join(PROJECT_ROOT, "scripts", "tmp_spec.png")

# %%
# ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    width_shift_range=0.1,
    zoom_range=0.1
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=16,
    subset='training',
    shuffle=True,
    seed=42,
    class_mode='categorical'
)
val_gen = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=16,
    subset='validation',
    shuffle=False,
    class_mode='categorical'
)

# Save class indices
joblib.dump(train_gen.class_indices, CLASS_INDICES_PATH)
print("✅ Class indices saved!")

# %%
# Build CNN with MobileNetV2
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True),
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
]

# %%
# Train CNN
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=callbacks
)
model.save(MODEL_SAVE_PATH)
print(f"✅ Model saved at: {MODEL_SAVE_PATH}")

# %%
# Evaluate
val_loss, val_acc = model.evaluate(val_gen)
print(f"Validation Accuracy: {val_acc:.2f}")

# %%
# Plot history
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title("Training Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title("Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# %%
# Confusion matrix
y_true = val_gen.classes
y_pred = np.argmax(model.predict(val_gen), axis=1)
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=val_gen.class_indices.keys())
disp.plot(xticks_rotation=45)
plt.title("CNN Confusion Matrix")
plt.show()

# %%
# Predict genre from image
def predict_genre_from_image(img_path, model, img_size=(224,224)):
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    pred_index = np.argmax(model.predict(img_array), axis=1)[0]
    index_to_genre = {v:k for k,v in joblib.load(CLASS_INDICES_PATH).items()}
    return index_to_genre[pred_index]

# Example
example_img = os.path.join(DATASET_PATH, "blues", "blues.00001.png")
print("Predicted genre:", predict_genre_from_image(example_img, model))

# %%
# Predict genre from audio
def predict_genre_from_audio(audio_file, model, duration=30, img_size=(224,224)):
    y, sr = librosa.load(audio_file, duration=duration)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(3,3))
    plt.axis('off')
    librosa.display.specshow(S_db, sr=sr, cmap='viridis')
    plt.savefig(TMP_SPEC_PATH, bbox_inches='tight', pad_inches=0)
    plt.close()

    genre = predict_genre_from_image(TMP_SPEC_PATH, model, img_size)
    os.remove(TMP_SPEC_PATH)
    return genre

# Example
audio_file = os.path.join(AUDIO_PATH, "blues", "blues.00001.wav")
print("Predicted genre from audio:", predict_genre_from_audio(audio_file, model))
