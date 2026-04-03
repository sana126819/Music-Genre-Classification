# visualize_predictions.py
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

class PredictionVisualizer:
    def __init__(self, model_path, dataset_path, img_size=(128,128), batch_size=1):
        """
        model_path: path to saved CNN model
        dataset_path: path to image dataset
        img_size: target size for images
        batch_size: batch size for ImageDataGenerator
        """
        self.model = tf.keras.models.load_model(model_path)
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size

        # Create validation generator
        datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)
        self.val_gen = datagen.flow_from_directory(
            dataset_path,
            target_size=img_size,
            batch_size=batch_size,
            subset='validation',
            shuffle=False
        )

        self.class_names = list(self.val_gen.class_indices.keys())

    def show_predictions(self):
        """Visualize one example per class with true vs predicted labels."""
        picked_classes = set()
        i = 0
        total_images = len(self.val_gen)
        
        while len(picked_classes) < len(self.class_names) and i < total_images:
            img, label = self.val_gen[i][0], self.val_gen[i][1]
            true_class = self.class_names[np.argmax(label[0])]

            if true_class not in picked_classes:
                pred_index = np.argmax(self.model.predict(img.reshape(1, *self.img_size, 3)))
                img_disp = (img[0] * 255).astype(np.uint8)
                
                plt.imshow(img_disp)
                plt.title(f"True: {true_class}, Pred: {self.class_names[pred_index]}")
                plt.axis('off')
                plt.show()

                picked_classes.add(true_class)
            i += 1


# Example usage
if __name__ == "__main__":
    visualizer = PredictionVisualizer(
        model_path="../models/cnn_model.keras",
        dataset_path="../dataset/image",
        img_size=(128,128),
        batch_size=1
    )
    visualizer.show_predictions()
