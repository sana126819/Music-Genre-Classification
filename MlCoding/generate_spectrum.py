''' import os
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

audio_path = "dataset/audio"
image_path = "dataset/image"
duration = 30  # seconds

os.makedirs(image_path, exist_ok=True)

for genre in os.listdir(audio_path):
    genre_dir = os.path.join(audio_path, genre)
    out_dir = os.path.join(image_path, genre)
    os.makedirs(out_dir, exist_ok=True)
    
    for fname in os.listdir(genre_dir):
        file_path = os.path.join(genre_dir, fname)
        if not file_path.lower().endswith((".wav", ".mp3")):
            continue
        try:
            y, sr = librosa.load(file_path, duration=duration)
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            plt.figure(figsize=(5,5))  # bigger images
            plt.axis('off')
            librosa.display.specshow(S_db, sr=sr, cmap='viridis')
            
            save_path = os.path.join(out_dir, fname.replace(".wav",".png").replace(".mp3",".png"))
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        except Exception as e:
            print(f"Skipped {fname}: {e}")

print("Spectrograms generated in dataset/image/")
'''
# generate_spectrum.py
import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def generate_spectrograms(audio_dir="dataset/audio",
                          image_dir="dataset/image",
                          duration=30,
                          img_size=(5,5)):
    """
    Convert audio files to spectrogram images.

    Parameters:
    - audio_dir: Path to input audio dataset (organized by genre folders)
    - image_dir: Path to save spectrogram images
    - duration: Duration (in seconds) to load from each audio file
    - img_size: Size of saved images (in inches, matplotlib figsize)
    """
    os.makedirs(image_dir, exist_ok=True)

    for genre in os.listdir(audio_dir):
        genre_dir = os.path.join(audio_dir, genre)
        out_dir = os.path.join(image_dir, genre)
        os.makedirs(out_dir, exist_ok=True)

        for fname in os.listdir(genre_dir):
            file_path = os.path.join(genre_dir, fname)
            if not file_path.lower().endswith((".wav", ".mp3")):
                continue
            try:
                # Load audio
                y, sr = librosa.load(file_path, duration=duration)

                # Generate Mel-spectrogram
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                S_db = librosa.power_to_db(S, ref=np.max)

                # Plot and save spectrogram
                plt.figure(figsize=img_size)
                plt.axis('off')
                librosa.display.specshow(S_db, sr=sr, cmap='viridis')

                save_path = os.path.join(out_dir, fname.replace(".wav", ".png").replace(".mp3", ".png"))
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()
            except Exception as e:
                print(f"Skipped {fname}: {e}")

    print(f"✅ Spectrograms generated in '{image_dir}'")

# Run as script
if __name__ == "__main__":
    generate_spectrograms()
