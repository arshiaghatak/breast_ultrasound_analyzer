# Breast Cancer Tumor Classifier

##This project is licensed under the MIT License. Please give credit to the author, Arshia Ghatak, when using or modifying this work. See the LICENSE file for details.

A machine learning model that classifies breast cancer tumors into three categories: **Benign**, **Normal**, and **Malignant** using multiple algorithms: Random Forest, SVM (Support Vector Machine), and MLP (Multi-Layer Perceptron).

## Results Summary

Based on the confusion matrix results with **balanced dataset** (70-30 train-test split + oversampling):

- **Overall Accuracy**: 84.3%
- **Best Performing Class**: **Normal** (93.9% accuracy)
- **Class-wise Performance**:
  - Normal: 93.9% accuracy
  - Malignant: 86.4% accuracy
  - Benign: 72.5% accuracy

## Dataset

The model was trained on **1,311 breast cancer images** (balanced through oversampling):
- **Benign**: 437 images (original)
- **Normal**: 437 images (133 original + 304 oversampled)
- **Malignant**: 437 images (210 original + 227 oversampled)

*Note: All MASK files were automatically excluded from training as requested. Dataset was balanced using oversampling to ensure equal representation of all classes.*

## Features

### Image Processing
- Automatic exclusion of mask files (files containing "MASK" in filename)
- Standardized image resizing to 224x224 pixels
- Multi-channel feature extraction (RGB and HSV color spaces)

### Feature Engineering
The model extracts 29 different features from each image:

1. **Color Features** (6 features):
   - Mean and standard deviation for R, G, B channels

2. **Texture Features** (3 features):
   - Gradient magnitude statistics
   - Texture energy

3. **Shape Features** (5 features):
   - Area, perimeter, aspect ratio
   - Circularity and compactness

4. **Color Histogram Features** (15 features):
   - HSV histogram bins (5 per channel)

## Files Structure

```
Polygence Folder/
├── breast_cancer_classifier.py    # Main training script
├── breast_cancer_gui.py          # GUI interface (recommended)
├── launch_gui.py                 # GUI launcher script
├── predict_new_image.py          # Command-line prediction script
├── requirements.txt              # Python dependencies
├── breast_cancer_model.pkl       # Trained Random Forest model
├── breast_cancer_model_svm.pkl   # Trained SVM model
├── breast_cancer_model_mlp.pkl   # Trained MLP model
├── scaler.pkl                    # Feature scaler (RF)
├── scaler_svm.pkl                # Feature scaler (SVM)
├── scaler_mlp.pkl                # Feature scaler (MLP)
├── confusion_matrix.png          # Confusion matrix visualization (RF)
├── confusion_matrix_svm.png      # Confusion matrix visualization (SVM)
├── confusion_matrix_mlp.png      # Confusion matrix visualization (MLP)
├── feature_importance.png        # Feature importance plot
└── README.md                     # This file
```

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python3 breast_cancer_classifier.py
```

### GUI Interface (Recommended)
Launch the user-friendly graphical interface:
```bash
python3 breast_cancer_gui.py
```
or
```bash
python3 launch_gui.py
```

**GUI Features:**
- **Image Selection**: Click "Select Image" to browse and choose an image
- **Image Preview**: See a thumbnail preview of your selected image
- **Model Selection**: Choose between Random Forest, SVM, or MLP classifiers
- **One-Click Prediction**: Click "Predict" to classify the image
- **Visual Results**: See confidence scores with progress bars
- ⚡ **Real-time Status**: Progress indicator and status updates

### Command Line Prediction
For command-line usage:
```bash
python3 predict_new_image.py
```
Then enter the path to the image you want to classify.

## Model Performance

### Confusion Matrix
The model shows **excellent balanced performance** across all classes after oversampling:
- **Normal**: 93.9% accuracy (best performing)
- **Malignant**: 86.4% accuracy  
- **Benign**: 72.5% accuracy

### Key Improvements with Oversampling
- **Overall Accuracy**: Improved from 66.0% to 84.3% (+18.3%)
- **Balanced Performance**: No single class dominates
- **Better Minority Recognition**: Normal and Malignant classes significantly improved

### Feature Importance
The most important features for classification include:
- Color statistics (mean RGB values)
- Texture characteristics
- Shape properties (area, circularity)

## Technical Details

- **Algorithms**: 
  - Random Forest Classifier (n_estimators=100)
  - SVM with RBF kernel (C=2.0, gamma='scale')
  - MLP Classifier (hidden_layers=(100, 50), max_iter=500, activation='relu', solver='adam')
- **Cross-validation**: 5-fold CV
- **Train-Test Split**: 70-30 with stratified sampling
- **Dataset Balancing**: Oversampling to equalize class representation
- **Feature Scaling**: StandardScaler applied to each model
- **Image Preprocessing**: Resize to 224x224, RGB conversion

## Important Notes

1. **Best Class**: The **Normal** class shows the highest accuracy (93.9%) in the balanced model
2. **Dataset Balancing**: Oversampling was used to ensure equal representation of all classes
3. **Mask Exclusion**: All files containing "MASK" are automatically filtered out
4. **Medical Disclaimer**: This is a research/educational tool and should not be used for actual medical diagnosis

## Visualizations

The training process generates several visualizations:
- `confusion_matrix.png`: Shows the classification performance matrix for Random Forest (in percentages)
- `confusion_matrix_svm.png`: Shows the classification performance matrix for SVM (in percentages)
- `confusion_matrix_mlp.png`: Shows the classification performance matrix for MLP (in percentages)
- `feature_importance.png`: Displays the top 10 most important features (Random Forest only)

**Note**: All confusion matrices display percentages (0-100%) instead of raw counts for easier interpretation. Each row sums to 100%, showing the distribution of predictions for each actual class.

## Model Persistence

The trained models and scalers are saved as:
- **Random Forest**: `breast_cancer_model.pkl` and `scaler.pkl`
- **SVM**: `breast_cancer_model_svm.pkl` and `scaler_svm.pkl`
- **MLP**: `breast_cancer_model_mlp.pkl` and `scaler_mlp.pkl`

This allows you to load and use any model for predictions without retraining. The GUI includes a model selector to switch between models on the fly.
