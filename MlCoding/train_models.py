# file: music_model_trainer.py

import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt


class FeatureManager:
    def __init__(self, models_path="./models"):
        self.models_path = models_path
        self.X = None
        self.y = None
        self.scaler = StandardScaler()

    def load_features(self):
        features_path = os.path.join(self.models_path, "features.npy")
        labels_path = os.path.join(self.models_path, "labels.npy")
        if not os.path.exists(features_path) or not os.path.exists(labels_path):
            raise FileNotFoundError("Features or labels not found in ./models")
        self.X = np.load(features_path)
        self.y = np.load(labels_path)
        print("✅ Features loaded:", self.X.shape, self.y.shape)

    def scale_features(self):
        self.X = self.scaler.fit_transform(self.X)
        print("✅ Features scaled.")

    def split_data(self, test_size=0.2, random_state=42):
        return train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )


class ModelTrainer:
    def __init__(self, model_class, param_grid, model_name="Model"):
        self.model_class = model_class
        self.param_grid = param_grid
        self.model_name = model_name
        self.best_model = None
        self.accuracy = None

    def train(self, X_train, y_train):
        print(f"⏳ Training {self.model_name} ...")
        grid = GridSearchCV(self.model_class(), self.param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        self.best_model = grid.best_estimator_
        print(f"✅ {self.model_name} Best Params:", grid.best_params_)

    def evaluate(self, X_test, y_test):
        self.accuracy = accuracy_score(y_test, self.best_model.predict(X_test))
        print(f"✅ {self.model_name} Accuracy:", self.accuracy)
        return self.accuracy

    def save_model(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, f"{self.model_name.lower().replace(' ', '_')}.joblib")
        joblib.dump(self.best_model, file_path)
        print(f"✅ {self.model_name} saved at {file_path}")


class EnsembleTrainer:
    def __init__(self, trained_models, voting="soft"):
        self.trained_models = trained_models
        self.voting = voting
        self.ensemble_model = None
        self.accuracy = None

    def train(self, X_train, y_train):
        # Prepare estimators for VotingClassifier
        estimators = [
            (m.model_name.lower().replace(" ", "_"), m.best_model) for m in self.trained_models
        ]
        self.ensemble_model = VotingClassifier(estimators=estimators, voting=self.voting)
        self.ensemble_model.fit(X_train, y_train)
        print("✅ Ensemble trained.")

    def evaluate(self, X_test, y_test):
        self.accuracy = accuracy_score(self.ensemble_model.predict(X_test), y_test)
        print("✅ Ensemble Accuracy:", self.accuracy)
        return self.accuracy


class AccuracyPlotter:
    @staticmethod
    def plot(models, accuracies):
        plt.bar(models, accuracies, color=["skyblue", "orange", "green", "purple"])
        plt.ylim(0, 1)
        plt.ylabel("Accuracy")
        plt.title("Music Genre Model Accuracies")
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.02, f"{acc:.2f}", ha='center', fontsize=12)
        plt.show()


# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    models_path = "./models"

    # 1. Load & scale features
    features = FeatureManager(models_path=models_path)
    features.load_features()
    features.scale_features()
    X_train, X_test, y_train, y_test = features.split_data()

    # 2. Train individual models
    rf_trainer = ModelTrainer(
        RandomForestClassifier,
        {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 5]},
        "Random Forest"
    )
    knn_trainer = ModelTrainer(
        KNeighborsClassifier,
        {'n_neighbors': [3, 5], 'weights': ['uniform', 'distance']},
        "KNN"
    )
    # IMPORTANT: Use probability=True for SVM so soft voting works
    svm_trainer = ModelTrainer(
        lambda: SVC(probability=True),  # use lambda to enable probability=True
        {'C': [0.1, 1], 'kernel': ['linear', 'rbf']},
        "SVM"
    )

    trainers = [rf_trainer, knn_trainer, svm_trainer]
    for t in trainers:
        t.train(X_train, y_train)
        t.evaluate(X_test, y_test)
        t.save_model(models_path)

    # 3. Ensemble
    ensemble_trainer = EnsembleTrainer(trainers)
    ensemble_trainer.train(X_train, y_train)
    ensemble_trainer.evaluate(X_test, y_test)

    # 4. Plot all accuracies
    AccuracyPlotter.plot(
        ["Random Forest", "KNN", "SVM", "Ensemble"],
        [rf_trainer.accuracy, knn_trainer.accuracy, svm_trainer.accuracy, ensemble_trainer.accuracy]
    )
