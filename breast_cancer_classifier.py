import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

class BreastCancerClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        # Secondary model: SVM (RBF kernel)
        self.svm_model = SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, random_state=42)
        self.svm_scaler = StandardScaler()
        # Third model: MLP (Multi-Layer Perceptron)
        self.mlp_model = MLPClassifier(hidden_layer_sizes=(128, 100, 75, 50, 25), max_iter=500, 
                                       activation='relu', solver='adam', random_state=42)
        self.mlp_scaler = StandardScaler()
        self.classes = ['Benign', 'Normal', 'Malignant']
        
    def load_images(self, folder_path):
        """Load images from a folder, excluding mask files"""
        images = []
        labels = []
        
        # Get all files in the folder
        files = os.listdir(folder_path)
        
        # Filter out mask files and only keep image files
        image_files = [f for f in files if 'mask' not in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Loading {len(image_files)} images from {os.path.basename(folder_path)}")
        
        for file in image_files:
            try:
                img_path = os.path.join(folder_path, file)
                img = Image.open(img_path).convert('RGB')
                
                # Resize image to standard size (224x224)
                img = img.resize((224, 224))
                img_array = np.array(img)
                
                images.append(img_array)
                
                # Extract label from folder name
                folder_name = os.path.basename(folder_path).lower()
                if 'benign' in folder_name:
                    labels.append(0)  # Benign
                elif 'normal' in folder_name:
                    labels.append(1)  # Normal
                elif 'malignant' in folder_name:
                    labels.append(2)  # Malignant
                    
            except Exception as e:
                print(f"Error loading {file}: {e}")
                continue
                
        return np.array(images), np.array(labels)
    
    def extract_features(self, images):
        """Extract features from images using multiple techniques"""
        features = []
        
        for img in images:
            # Convert to grayscale for some features
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # 1. Basic statistical features
            mean_r, mean_g, mean_b = np.mean(img, axis=(0, 1))
            std_r, std_g, std_b = np.std(img, axis=(0, 1))
            
            # 2. Texture features using LBP-like approach
            texture_features = self.extract_texture_features(gray)
            
            # 3. Shape features
            shape_features = self.extract_shape_features(gray)
            
            # 4. Color histogram features
            color_features = self.extract_color_features(img)
            
            # Combine all features
            combined_features = [
                mean_r, mean_g, mean_b,
                std_r, std_g, std_b,
                *texture_features,
                *shape_features,
                *color_features
            ]
            
            features.append(combined_features)
            
        return np.array(features)
    
    def extract_texture_features(self, gray_img):
        """Extract texture features using Local Binary Pattern approximation"""
        # Calculate gradient magnitude
        grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Texture statistics
        texture_mean = np.mean(gradient_magnitude)
        texture_std = np.std(gradient_magnitude)
        texture_energy = np.sum(gradient_magnitude**2)
        
        return [texture_mean, texture_std, texture_energy]
    
    def extract_shape_features(self, gray_img):
        """Extract shape-related features"""
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Calculate shape features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            # Aspect ratio
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Circularity
            circularity = 4 * np.pi * area / (perimeter**2) if perimeter > 0 else 0
            
            # Compactness
            compactness = area / (perimeter**2) if perimeter > 0 else 0
            
        else:
            area = perimeter = aspect_ratio = circularity = compactness = 0
            
        return [area, perimeter, aspect_ratio, circularity, compactness]
    
    def extract_color_features(self, img):
        """Extract color histogram features"""
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # Calculate histogram for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
        
        # Normalize histograms
        hist_h = hist_h.flatten() / hist_h.sum()
        hist_s = hist_s.flatten() / hist_s.sum()
        hist_v = hist_v.flatten() / hist_v.sum()
        
        # Return top 5 bins from each channel
        return list(hist_h[:5]) + list(hist_s[:5]) + list(hist_v[:5])
    
    def balance_dataset(self, images, labels):
        """Balance the dataset by oversampling minority classes"""
        print("\nBalancing dataset...")
        
        # Count samples per class
        unique_classes, counts = np.unique(labels, return_counts=True)
        print("Original class distribution:")
        for i, class_name in enumerate(self.classes):
            if i in unique_classes:
                idx = np.where(unique_classes == i)[0][0]
                print(f"  {class_name}: {counts[idx]} samples")
        
        # Find the maximum count (majority class)
        max_count = np.max(counts)
        print(f"\nTarget samples per class: {max_count}")
        
        # Oversample each class to match the majority class
        balanced_images = []
        balanced_labels = []
        
        for class_idx in unique_classes:
            # Get indices for current class
            class_indices = np.where(labels == class_idx)[0]
            class_images = images[class_indices]
            class_labels = labels[class_indices]
            
            # If this class has fewer samples than the majority, oversample
            if len(class_images) < max_count:
                # Calculate how many samples to add
                n_samples_to_add = max_count - len(class_images)
                
                # Randomly sample with replacement to get the additional samples
                additional_indices = np.random.choice(len(class_images), 
                                                   size=n_samples_to_add, 
                                                   replace=True)
                
                # Add original samples
                balanced_images.extend(class_images)
                balanced_labels.extend(class_labels)
                
                # Add oversampled samples
                balanced_images.extend(class_images[additional_indices])
                balanced_labels.extend(class_labels[additional_indices])
                
                print(f"  {self.classes[class_idx]}: {len(class_images)} → {max_count} (added {n_samples_to_add})")
            else:
                # If class already has enough samples, just add them
                balanced_images.extend(class_images)
                balanced_labels.extend(class_labels)
                print(f"  {self.classes[class_idx]}: {len(class_images)} (no change needed)")
        
        balanced_images = np.array(balanced_images)
        balanced_labels = np.array(balanced_labels)
        
        print(f"\nBalanced dataset size: {len(balanced_images)} samples")
        return balanced_images, balanced_labels
    
    def train_model(self, X_train, y_train):
        """Train the Random Forest model"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model and return predictions"""
        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nTest Accuracy: {accuracy:.3f}")
        
        return y_pred, accuracy
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize to percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_percent,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=self.classes,
            yticklabels=self.classes,
            vmin=0,
            vmax=100,
            annot_kws={"size": 16, "weight": "bold"},
            square=True,
            linewidths=0.5,
            linecolor='white'
        )
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14, rotation=0)
        plt.title('Confusion Matrix - Random Forest (Percentages)')
        plt.xlabel('Predicted', fontsize=18, fontweight='bold')
        plt.ylabel('Actual', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/Users/arshia/Desktop/Polygence Folder/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.classes))
        
        return cm

    def plot_confusion_matrix_svm(self, y_true, y_pred):
        """Plot confusion matrix for SVM"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize to percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_percent,
            annot=True,
            fmt='.1f',
            cmap='Purples',
            xticklabels=self.classes,
            yticklabels=self.classes,
            vmin=0,
            vmax=100,
            annot_kws={"size": 16, "weight": "bold"},
            square=True,
            linewidths=0.5,
            linecolor='white'
        )
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14, rotation=0)
        plt.title('Confusion Matrix - SVM (Percentages)')
        plt.xlabel('Predicted', fontsize=18, fontweight='bold')
        plt.ylabel('Actual', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/Users/arshia/Desktop/Polygence Folder/confusion_matrix_svm.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nClassification Report (SVM):")
        print(classification_report(y_true, y_pred, target_names=self.classes))
        
        return cm
    
    def plot_confusion_matrix_mlp(self, y_true, y_pred):
        """Plot confusion matrix for MLP"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize to percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_percent,
            annot=True,
            fmt='.1f',
            cmap='Greens',
            xticklabels=self.classes,
            yticklabels=self.classes,
            vmin=0,
            vmax=100,
            annot_kws={"size": 16, "weight": "bold"},
            square=True,
            linewidths=0.5,
            linecolor='white'
        )
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14, rotation=0)
        plt.title('Confusion Matrix - MLP (Percentages)')
        plt.xlabel('Predicted', fontsize=18, fontweight='bold')
        plt.ylabel('Actual', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.savefig('/Users/arshia/Desktop/Polygence Folder/confusion_matrix_mlp.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\nClassification Report (MLP):")
        print(classification_report(y_true, y_pred, target_names=self.classes))
        
        return cm
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        feature_names = [
            'Mean_R', 'Mean_G', 'Mean_B', 'Std_R', 'Std_G', 'Std_B',
            'Texture_Mean', 'Texture_Std', 'Texture_Energy',
            'Area', 'Perimeter', 'Aspect_Ratio', 'Circularity', 'Compactness',
            'H_1', 'H_2', 'H_3', 'H_4', 'H_5',
            'S_1', 'S_2', 'S_3', 'S_4', 'S_5',
            'V_1', 'V_2', 'V_3', 'V_4', 'V_5'
        ]
        
        importances = self.model.feature_importances_
        
        # Get top 10 most important features
        top_indices = np.argsort(importances)[::-1][:10]
        
        # Get bottom 10 least important features
        bottom_indices = np.argsort(importances)[:10]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Top 10 features plot
        bars1 = ax1.bar(range(10), importances[top_indices], color='#1a4577', edgecolor='white', linewidth=1.5)
        ax1.set_xticks(range(10))
        ax1.set_xticklabels([feature_names[i] for i in top_indices], rotation=45, ha='right')
        ax1.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
        ax1.set_title('Top 10 Feature Importances (out of 29 total features)', fontsize=14, fontweight='bold', pad=15)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on top 10 bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Bottom 10 features plot
        bars2 = ax2.bar(range(10), importances[bottom_indices], color='#8b2635', edgecolor='white', linewidth=1.5)
        ax2.set_xticks(range(10))
        ax2.set_xticklabels([feature_names[i] for i in bottom_indices], rotation=45, ha='right')
        ax2.set_ylabel('Importance Score', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Feature', fontsize=12, fontweight='bold')
        ax2.set_title('Bottom 10 Feature Importances (out of 29 total features)', fontsize=14, fontweight='bold', pad=15)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bottom 10 bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig('/Users/arshia/Desktop/Polygence Folder/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

    def predict_weighted_vote(
        self,
        features,
        models=None,
        weights=None,
        class_names=None
    ):
        """
        Combine predictions from the available models using weighted voting.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix with shape (n_samples, n_features). If a single
            sample is provided, the array can be 1-D; it will be reshaped.
        models : dict, optional
            Dictionary with model configurations. Each entry should be of the form
            {
                'rf': {'model': RandomForestClassifier, 'scaler': StandardScaler},
                'svm': {'model': SVC, 'scaler': StandardScaler},
                'mlp': {'model': MLPClassifier, 'scaler': StandardScaler}
            }
            If omitted, the method falls back to the estimator and scaler
            attributes stored on the classifier instance.
        weights : dict, optional
            Mapping from model keys to non-normalized weights. Defaults to a
            bias towards Random Forest, followed by MLP, with SVM receiving the
            smallest contribution.
        class_names : list[str], optional
            Ordered list of class names. If not provided, the classifier's
            internal `self.classes` definition is used (when available).

        Returns
        -------
        tuple[int, np.ndarray, list[dict]] | None
            - index of the predicted class
            - combined probability scores
            - per-model contribution details

            Returns None if no compatible models are available.
        """
        if features is None:
            raise ValueError("Expected feature array, received None.")

        features = np.atleast_2d(features)
        class_names = class_names or getattr(self, "classes", None)

        default_weights = {'rf': 0.6, 'mlp': 0.25, 'svm': 0.15}
        model_weights = weights or default_weights

        if models is None:
            models = {
                'rf': {'model': getattr(self, 'model', None), 'scaler': getattr(self, 'scaler', None)},
                'svm': {'model': getattr(self, 'svm_model', None), 'scaler': getattr(self, 'svm_scaler', None)},
                'mlp': {'model': getattr(self, 'mlp_model', None), 'scaler': getattr(self, 'mlp_scaler', None)},
            }

        available_models = []
        for key, config in models.items():
            model = config.get('model')
            scaler = config.get('scaler')

            if model is None or scaler is None:
                continue

            try:
                transformed = scaler.transform(features)
                if not hasattr(model, "predict_proba"):
                    raise AttributeError("Model does not implement predict_proba")
                probabilities = model.predict_proba(transformed)[0]
            except Exception as error:
                print(f"Warning: unable to obtain probabilities for {key.upper()} model: {error}")
                continue

            predicted_class = int(np.argmax(probabilities))
            weight = model_weights.get(key, 0)
            if weight <= 0:
                continue

            available_models.append({
                'key': key,
                'probabilities': probabilities,
                'prediction': predicted_class,
                'weight': weight
            })

        if not available_models:
            return None

        total_weight = sum(model['weight'] for model in available_models)
        if total_weight == 0:
            equal_weight = 1.0 / len(available_models)
            for model in available_models:
                model['weight'] = equal_weight
        else:
            for model in available_models:
                model['weight'] = model['weight'] / total_weight

        combined_scores = np.zeros_like(available_models[0]['probabilities'])
        for model in available_models:
            combined_scores += model['weight'] * model['probabilities']

        final_prediction = int(np.argmax(combined_scores))

        model_names = {
            'rf': 'Random Forest',
            'svm': 'SVM',
            'mlp': 'MLP'
        }

        detailed_results = []
        for model in available_models:
            detailed_results.append({
                'name': model_names.get(model['key'], model['key'].upper()),
                'weight': model['weight'],
                'prediction': model['prediction'],
                'probabilities': model['probabilities']
            })

        return final_prediction, combined_scores, detailed_results

    def plot_confusion_matrix_weighted(self, y_true, y_pred):
        """Plot confusion matrix for the weighted ensemble"""
        cm = confusion_matrix(y_true, y_pred)
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm_percent,
            annot=True,
            fmt='.1f',
            cmap='OrRd',
            xticklabels=self.classes,
            yticklabels=self.classes,
            vmin=0,
            vmax=100,
            annot_kws={"size": 16, "weight": "bold"},
            square=True,
            linewidths=0.5,
            linecolor='white'
        )
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14, rotation=0)
        plt.title('Confusion Matrix - Weighted Ensemble (Percentages)')
        plt.xlabel('Predicted', fontsize=18, fontweight='bold')
        plt.ylabel('Actual', fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.tight_layout()
        plt.savefig('/Users/arshia/Desktop/Polygence Folder/confusion_matrix_weighted.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("Breast Cancer Tumor Classifier")
    print("=" * 40)
    
    # Initialize classifier
    classifier = BreastCancerClassifier()
    
    # Define paths
    base_path = "/Users/arshia/Desktop"
    benign_path = os.path.join(base_path, "benign")
    normal_path = os.path.join(base_path, "normal")
    malignant_path = os.path.join(base_path, "malignant")
    
    # Load images from all three classes
    print("Loading images...")
    benign_images, benign_labels = classifier.load_images(benign_path)
    normal_images, normal_labels = classifier.load_images(normal_path)
    malignant_images, malignant_labels = classifier.load_images(malignant_path)
    
    # Combine all data
    all_images = np.vstack([benign_images, normal_images, malignant_images])
    all_labels = np.hstack([benign_labels, normal_labels, malignant_labels])
    
    print(f"\nTotal images loaded: {len(all_images)}")
    print(f"Benign: {len(benign_images)}, Normal: {len(normal_images)}, Malignant: {len(malignant_images)}")
    
    # Balance the dataset by oversampling
    balanced_images, balanced_labels = classifier.balance_dataset(all_images, all_labels)
    
    # Extract features from balanced dataset
    print("\nExtracting features from balanced dataset...")
    features = classifier.extract_features(balanced_images)
    print(f"Feature matrix shape: {features.shape}")
    
    # Split data with 70-30 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, balanced_labels, test_size=0.3, random_state=42, stratify=balanced_labels
    )
    
    print(f"\nTrain-Test Split (70-30):")
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(features)*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(features)*100:.1f}%)")
    
    # Show class distribution in train and test sets
    print(f"\nBalanced Class Distribution:")
    for i, class_name in enumerate(classifier.classes):
        train_count = np.sum(y_train == i)
        test_count = np.sum(y_test == i)
        print(f"{class_name:10}: Train={train_count:3d}, Test={test_count:3d}")
    
    # Train model
    print("\nTraining Random Forest model...")
    classifier.train_model(X_train, y_train)
    
    # Evaluate model (Random Forest)
    print("\nEvaluating model...")
    y_pred, accuracy = classifier.evaluate_model(X_test, y_test)
    
    # Plot confusion matrix (RF)
    cm = classifier.plot_confusion_matrix(y_test, y_pred)
    
    # Plot feature importance
    classifier.plot_feature_importance()
    
    # Find best performing class
    class_accuracy = []
    for i, class_name in enumerate(classifier.classes):
        class_mask = (y_test == i)
        if np.sum(class_mask) > 0:
            class_acc = np.sum((y_test == i) & (y_pred == i)) / np.sum(class_mask)
            class_accuracy.append(class_acc)
            print(f"{class_name} accuracy: {class_acc:.3f}")
    
    best_class_idx = np.argmax(class_accuracy)
    best_class = classifier.classes[best_class_idx]
    print(f"\nBest performing class: {best_class} (accuracy: {class_accuracy[best_class_idx]:.3f})")
    
    # Train and evaluate SVM
    print("\nTraining SVM model...")
    X_train_svm = classifier.svm_scaler.fit_transform(X_train)
    classifier.svm_model.fit(X_train_svm, y_train)
    
    print("\nEvaluating SVM model...")
    X_test_svm = classifier.svm_scaler.transform(X_test)
    y_pred_svm = classifier.svm_model.predict(X_test_svm)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    print(f"SVM Test Accuracy: {svm_accuracy:.3f}")
    classifier.plot_confusion_matrix_svm(y_test, y_pred_svm)
    
    # Train and evaluate MLP
    print("\nTraining MLP model...")
    X_train_mlp = classifier.mlp_scaler.fit_transform(X_train)
    classifier.mlp_model.fit(X_train_mlp, y_train)
    
    print("\nEvaluating MLP model...")
    X_test_mlp = classifier.mlp_scaler.transform(X_test)
    y_pred_mlp = classifier.mlp_model.predict(X_test_mlp)
    mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
    print(f"MLP Test Accuracy: {mlp_accuracy:.3f}")
    classifier.plot_confusion_matrix_mlp(y_test, y_pred_mlp)
    
    # Evaluate weighted ensemble if at least one model is available
    ensemble_predictions = []
    ensemble_scores = []
    ensemble_models = {
        'rf': {'model': classifier.model, 'scaler': classifier.scaler},
        'svm': {'model': classifier.svm_model, 'scaler': classifier.svm_scaler},
        'mlp': {'model': classifier.mlp_model, 'scaler': classifier.mlp_scaler},
    }

    for sample in X_test:
        result = classifier.predict_weighted_vote(sample, models=ensemble_models)
        if result is None:
            raise ValueError("Weighted ensemble could not be evaluated because no models produced predictions.")
        final_pred, combined_scores, _ = result
        ensemble_predictions.append(final_pred)
        ensemble_scores.append(combined_scores)

    ensemble_predictions = np.array(ensemble_predictions)
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
    print(f"\nWeighted Ensemble Test Accuracy: {ensemble_accuracy:.3f}")
    classifier.plot_confusion_matrix_weighted(y_test, ensemble_predictions)

    print("\nWeighted Ensemble Classification Report:")
    print(classification_report(y_test, ensemble_predictions, target_names=classifier.classes))

    # Simple comparison summary
    print("\nModel Comparison Summary:")
    print(f"- Random Forest Accuracy: {accuracy:.3f}")
    print(f"- SVM Accuracy         : {svm_accuracy:.3f}")
    print(f"- MLP Accuracy         : {mlp_accuracy:.3f}")
    print(f"- Weighted Ensemble    : {ensemble_accuracy:.3f}")
    
    # Extract and display weighted ensemble weights
    default_weights = {'rf': 0.6, 'mlp': 0.25, 'svm': 0.15}
    model_names = {'rf': 'Random Forest', 'svm': 'SVM', 'mlp': 'MLP'}
    
    # Get normalized weights by making a sample prediction
    sample_result = classifier.predict_weighted_vote(X_test[0:1], models=ensemble_models, weights=default_weights)
    if sample_result is not None:
        _, _, model_details = sample_result
        
        # Extract normalized weights from model details
        weights_dict = {detail['name']: detail['weight'] for detail in model_details}
        
        print("\n" + "=" * 50)
        print("Weighted Ensemble Model Weights:")
        print("=" * 50)
        for model_name, weight in weights_dict.items():
            print(f"{model_name:15}: {weight*100:5.1f}%")
        print("=" * 50)
        
        # Plot the weights
        fig, ax = plt.subplots(figsize=(10, 6))
        models_list = list(weights_dict.keys())
        weights_list = [weights_dict[m] * 100 for m in models_list]
        colors = ['#1a4577', '#8b2635', '#2c5c91']  # Matching GUI colors
        
        bars = ax.bar(models_list, weights_list, color=colors, edgecolor='white', linewidth=2)
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights_list):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{weight:.1f}%',
                   ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_ylabel('Weight (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax.set_title('Weighted Ensemble Model Weights', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, max(weights_list) * 1.2)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig('/Users/arshia/Desktop/Polygence Folder/ensemble_weights.png', dpi=300, bbox_inches='tight')
        print("\nEnsemble weights plot saved to: ensemble_weights.png")
        plt.close()
    
    # Save models
    import joblib
    joblib.dump(classifier.model, '/Users/arshia/Desktop/Polygence Folder/breast_cancer_model.pkl')
    joblib.dump(classifier.scaler, '/Users/arshia/Desktop/Polygence Folder/scaler.pkl')
    joblib.dump(classifier.svm_model, '/Users/arshia/Desktop/Polygence Folder/breast_cancer_model_svm.pkl')
    joblib.dump(classifier.svm_scaler, '/Users/arshia/Desktop/Polygence Folder/scaler_svm.pkl')
    joblib.dump(classifier.mlp_model, '/Users/arshia/Desktop/Polygence Folder/breast_cancer_model_mlp.pkl')
    joblib.dump(classifier.mlp_scaler, '/Users/arshia/Desktop/Polygence Folder/scaler_mlp.pkl')
    print("\nModels saved successfully!")

if __name__ == "__main__":
    main()
