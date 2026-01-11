import os
import numpy as np
import cv2
from PIL import Image
import joblib
from breast_cancer_classifier import BreastCancerClassifier

def predict_single_image(image_path, model_path, scaler_path):
    """
    Predict the class of a single breast cancer image
    
    Args:
        image_path: Path to the image file
        model_path: Path to the saved Random Forest model
        scaler_path: Path to the saved scaler
    
    Returns:
        prediction: Predicted class (0=Benign, 1=Normal, 2=Malignant)
        confidence: Prediction confidence scores
    """
    
    # Load the trained model and scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Initialize classifier for feature extraction
    classifier = BreastCancerClassifier()
    
    # Load and preprocess the image
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        
        # Extract features (same as training)
        features = classifier.extract_features(np.array([img_array]))
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        confidence_scores = model.predict_proba(features_scaled)[0]
        
        return prediction, confidence_scores
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None

def main():
    # Paths to the trained model and scaler
    model_path = '/Users/arshia/Desktop/Polygence Folder/breast_cancer_model.pkl'
    scaler_path = '/Users/arshia/Desktop/Polygence Folder/scaler.pkl'
    
    # Class names
    classes = ['Benign', 'Normal', 'Malignant']
    
    # Example usage - you can change this path to any image you want to test
    test_image_path = input("Enter the path to the image you want to classify: ")
    
    if not os.path.exists(test_image_path):
        print("Image file not found!")
        return
    
    # Make prediction
    prediction, confidence = predict_single_image(test_image_path, model_path, scaler_path)
    
    if prediction is not None:
        print(f"\nPrediction Results:")
        print(f"Predicted Class: {classes[prediction]}")
        print(f"\nConfidence Scores:")
        for i, class_name in enumerate(classes):
            print(f"{class_name}: {confidence[i]:.3f}")
        
        # Find the most confident prediction
        max_confidence_idx = np.argmax(confidence)
        print(f"\nMost Confident Prediction: {classes[max_confidence_idx]} ({confidence[max_confidence_idx]:.3f})")
    else:
        print("Failed to make prediction")

if __name__ == "__main__":
    main()
