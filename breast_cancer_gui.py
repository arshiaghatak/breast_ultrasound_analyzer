import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import numpy as np
import cv2
from PIL import Image, ImageTk
import joblib
from breast_cancer_classifier import BreastCancerClassifier
import threading

class BreastCancerPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Breast Cancer Tumor Classifier")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')

        # Configure widget styles for consistent contrast
        self.style = ttk.Style(self.root)
        if self.style.theme_use() != "clam":
            self.style.theme_use("clam")
        self._configure_styles()
        
        # Model and scaler paths
        self.model_path = '/Users/arshia/Desktop/Polygence Folder/breast_cancer_model.pkl'
        self.scaler_path = '/Users/arshia/Desktop/Polygence Folder/scaler.pkl'
        
        # Class names
        self.classes = ['Benign', 'Normal', 'Malignant']
        
        # Setup model weights for weighted voting (higher weight favors model)
        self.model_weights = {
            'rf': 0.6,   # Random Forest has the highest influence
            'mlp': 0.25,
            'svm': 0.15
        }
        
        # Initialize variables
        self.selected_image_path = None
        self.model = None
        self.scaler = None
        self.classifier = None
        self.selected_model = 'rf'  # Default: Random Forest
        self.models = {
            'rf': {'model': None, 'scaler': None, 'path': '/Users/arshia/Desktop/Polygence Folder/breast_cancer_model.pkl', 'scaler_path': '/Users/arshia/Desktop/Polygence Folder/scaler.pkl'},
            'svm': {'model': None, 'scaler': None, 'path': '/Users/arshia/Desktop/Polygence Folder/breast_cancer_model_svm.pkl', 'scaler_path': '/Users/arshia/Desktop/Polygence Folder/scaler_svm.pkl'},
            'mlp': {'model': None, 'scaler': None, 'path': '/Users/arshia/Desktop/Polygence Folder/breast_cancer_model_mlp.pkl', 'scaler_path': '/Users/arshia/Desktop/Polygence Folder/scaler_mlp.pkl'}
        }
        
        # Load models and scalers
        self.load_models()
        
        # Create GUI elements
        self.create_widgets()
        
    def _configure_styles(self):
        """Configure ttk styles to ensure button contrast in all states"""
        primary_bg = '#1a4577'
        primary_active = '#0d2744'
        primary_disabled = '#2c5c91'

        danger_bg = '#8b2635'
        danger_active = '#6b1e29'
        danger_disabled = '#b04a59'

        button_font = ("Arial", 14, "bold")

        self.style.configure(
            "Primary.TButton",
            font=button_font,
            background=primary_bg,
            foreground='#FFFFFF',
            padding=(20, 10),
            borderwidth=0
        )
        self.style.map(
            "Primary.TButton",
            background=[("active", primary_active), ("disabled", primary_disabled)],
            foreground=[("disabled", '#FFFFFF'), ("active", '#FFFFFF')]
        )

        self.style.configure(
            "Danger.TButton",
            font=button_font,
            background=danger_bg,
            foreground='#FFFFFF',
            padding=(20, 10),
            borderwidth=0
        )
        self.style.map(
            "Danger.TButton",
            background=[("active", danger_active), ("disabled", danger_disabled)],
            foreground=[("disabled", '#FFFFFF'), ("active", '#FFFFFF')]
        )

    def load_models(self):
        """Load all trained models and scalers"""
        try:
            self.classifier = BreastCancerClassifier()
            
            # Load Random Forest model
            self.models['rf']['model'] = joblib.load(self.models['rf']['path'])
            self.models['rf']['scaler'] = joblib.load(self.models['rf']['scaler_path'])
            
            # Load SVM model
            try:
                self.models['svm']['model'] = joblib.load(self.models['svm']['path'])
                self.models['svm']['scaler'] = joblib.load(self.models['svm']['scaler_path'])
            except FileNotFoundError:
                print("SVM model not found, continuing without it")
            
            # Load MLP model
            try:
                self.models['mlp']['model'] = joblib.load(self.models['mlp']['path'])
                self.models['mlp']['scaler'] = joblib.load(self.models['mlp']['scaler_path'])
            except FileNotFoundError:
                print("MLP model not found, continuing without it")
            
            # Set default model and scaler
            self.model = self.models['rf']['model']
            self.scaler = self.models['rf']['scaler']
            
            print("Models loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            
    def create_widgets(self):
        """Create all GUI widgets"""
        # Title
        title_label = tk.Label(self.root, text="Breast Cancer Tumor Classifier", 
                              font=("Arial", 20, "bold"), bg='#f0f0f0', fg='#2c3e50')
        title_label.pack(pady=20)
        
        # Subtitle
        subtitle_label = tk.Label(self.root, text="Select an image to classify as Benign, Normal, or Malignant", 
                                 font=("Arial", 12), bg='#f0f0f0', fg='#7f8c8d')
        subtitle_label.pack(pady=(0, 10))
        
        # Model selection frame
        model_frame = tk.Frame(self.root, bg='#f0f0f0')
        model_frame.pack(pady=10)
        
        tk.Label(model_frame, text="Select Model:", font=("Arial", 12), 
                bg='#f0f0f0', fg='#2c3e50').pack(side='left', padx=5)
        
        self.model_var = tk.StringVar(value='Random Forest')
        model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                     values=['Random Forest', 'SVM', 'MLP'],
                                     state='readonly', width=20, font=("Arial", 12))
        model_dropdown.pack(side='left', padx=5)
        model_dropdown.bind('<<ComboboxSelected>>', self.on_model_change)
        
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Left frame for image selection and display
        left_frame = tk.Frame(main_frame, bg='#f0f0f0')
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Image selection frame
        selection_frame = tk.LabelFrame(left_frame, text="Image Selection", 
                                       font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#2c3e50')
        selection_frame.pack(fill='x', pady=(0, 20))
        
        # Select image button - darker background for better contrast
        select_btn = ttk.Button(selection_frame, text="Select Image",
                                command=self.select_image, style="Primary.TButton",
                                cursor='hand2')
        select_btn.pack(pady=10)
        
        # Selected image path label
        self.path_label = tk.Label(selection_frame, text="No image selected", 
                                  font=("Arial", 10), bg='#f0f0f0', fg='#7f8c8d',
                                  wraplength=400)
        self.path_label.pack(pady=5)
        
        # Image preview frame
        preview_frame = tk.LabelFrame(left_frame, text="Image Preview", 
                                     font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#2c3e50')
        preview_frame.pack(fill='both', expand=True)
        
        # Image display label
        self.image_label = tk.Label(preview_frame, text="No image selected", 
                                   font=("Arial", 12), bg='#f0f0f0', fg='#7f8c8d')
        self.image_label.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Right frame for prediction results
        right_frame = tk.Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side='right', fill='both', padx=(10, 0))
        
        # Prediction frame
        prediction_frame = tk.LabelFrame(right_frame, text="Prediction Results", 
                                        font=("Arial", 12, "bold"), bg='#f0f0f0', fg='#2c3e50')
        prediction_frame.pack(fill='both', expand=True)
        
        # Predict button - darker background for better contrast
        self.predict_btn = ttk.Button(prediction_frame, text="Predict",
                                      command=self.predict_image, style="Danger.TButton",
                                      cursor='hand2', state='disabled')
        self.predict_btn.pack(pady=10)
        
        # Results display
        self.results_text = tk.Text(prediction_frame, height=15, width=30, 
                                   font=("Arial", 10), bg='#ffffff', fg='#2c3e50',
                                   wrap='word', state='disabled')
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(prediction_frame, mode='indeterminate')
        self.progress.pack(fill='x', padx=10, pady=5)
        
        # Status bar
        self.status_label = tk.Label(self.root, text="Ready", 
                                    font=("Arial", 10), bg='#f0f0f0', fg='#7f8c8d',
                                    relief='sunken', anchor='w')
        self.status_label.pack(side='bottom', fill='x', padx=10, pady=5)
        
    def on_model_change(self, event):
        """Handle model selection change"""
        selected = self.model_var.get()
        model_map = {
            'Random Forest': 'rf',
            'SVM': 'svm',
            'MLP': 'mlp'
        }
        
        self.selected_model = model_map[selected]
        
        # Update active model and scaler if available
        if self.models[self.selected_model]['model'] is not None:
            self.model = self.models[self.selected_model]['model']
            self.scaler = self.models[self.selected_model]['scaler']
            self.update_status(f"Switched to {selected} model")
        else:
            messagebox.showwarning("Warning", f"{selected} model not found. Please train the model first.")
            # Revert to previous selection
            self.model_var.set('Random Forest')
            self.selected_model = 'rf'
            self.model = self.models['rf']['model']
            self.scaler = self.models['rf']['scaler']
        
    def select_image(self):
        """Open file dialog to select an image"""
        filetypes = [
            ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff"),
            ("PNG files", "*.png"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select an image to classify",
            filetypes=filetypes
        )
        
        if filename:
            self.selected_image_path = filename
            self.path_label.config(text=f"Selected: {os.path.basename(filename)}")
            self.predict_btn.config(state='normal')
            self.display_image_preview(filename)
            self.update_status(f"Image selected: {os.path.basename(filename)}")
            
    def display_image_preview(self, image_path):
        """Display a preview of the selected image"""
        try:
            # Load and resize image for preview
            img = Image.open(image_path)
            
            # Calculate size for preview (max 300x300)
            img.thumbnail((300, 300), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Update image label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
        except Exception as e:
            self.image_label.config(image="", text="Error loading image preview")
            messagebox.showerror("Error", f"Failed to load image preview: {str(e)}")
            
    def predict_image(self):
        """Predict the class of the selected image"""
        if not self.selected_image_path:
            messagebox.showwarning("Warning", "Please select an image first!")
            return
            
        # Start prediction in a separate thread to prevent GUI freezing
        threading.Thread(target=self._predict_image_thread, daemon=True).start()
        
    def _predict_image_thread(self):
        """Prediction logic running in separate thread"""
        try:
            # Update GUI to show progress
            self.root.after(0, self._start_prediction)
            
            # Load and preprocess the image
            img = Image.open(self.selected_image_path).convert('RGB')
            img = img.resize((224, 224))
            img_array = np.array(img)
            
            # Extract features
            features = self.classifier.extract_features(np.array([img_array]))
            
            weighted_result = self._weighted_vote(features)

            if weighted_result is None:
                raise ValueError("No models available for prediction. Please ensure at least one trained model is present.")

            final_prediction, final_scores, model_details = weighted_result
            
            # Update GUI with results
            self.root.after(0, lambda: self._display_results(final_prediction, final_scores, model_details))
            
        except Exception as e:
            self.root.after(0, lambda: self._prediction_error(str(e)))
            
    def _start_prediction(self):
        """Update GUI when prediction starts"""
        self.predict_btn.config(state='disabled', text="Predicting...")
        self.progress.start()
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Analyzing image...\nPlease wait...")
        self.results_text.config(state='disabled')
        self.update_status("Predicting...")
        
    def _weighted_vote(self, features):
        """Combine predictions from all available models using weighted voting"""
        return self.classifier.predict_weighted_vote(
            features,
            models=self.models,
            weights=self.model_weights,
            class_names=self.classes
        )

    def _display_results(self, prediction, confidence_scores, model_details):
        """Display prediction results in GUI"""
        # Stop progress bar
        self.progress.stop()
        
        # Enable predict button
        self.predict_btn.config(state='normal', text="Predict")
        
        # Clear and update results
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        
        # Format results
        results = "🎯 PREDICTION RESULTS (Weighted Voting)\n"
        results += "=" * 30 + "\n\n"
        
        results += f"📊 Final Decision:\n"
        results += f"   {self.classes[prediction]} ({confidence_scores[prediction]:.1%})\n\n"

        results += "🤖 Model Contributions:\n"
        for detail in model_details:
            model_name = detail['name']
            weight_percentage = detail['weight'] * 100
            predicted_class = self.classes[detail['prediction']]
            confidence = detail['probabilities'][detail['prediction']]
            results += f"   {model_name:14} → {predicted_class:10} ({confidence:.1%}) | Weight: {weight_percentage:.0f}%\n"
        results += "\n"
        
        results += "📈 Confidence Scores:\n"
        for i, class_name in enumerate(self.classes):
            percentage = confidence_scores[i] * 100
            bar_length = int(percentage / 2)  # Scale to 50 chars max
            bar = "█" * bar_length + "░" * (50 - bar_length)
            results += f"   {class_name:10}: {percentage:5.1f}% [{bar}]\n"
        
        results += f"\n🏆 Most Confident:\n"
        max_idx = np.argmax(confidence_scores)
        results += f"   {self.classes[max_idx]} ({confidence_scores[max_idx]:.1%})\n"
        
        # Color coding for prediction confidence
        if confidence_scores[prediction] > 0.7:
            results += "\n✅ High confidence prediction"
        elif confidence_scores[prediction] > 0.5:
            results += "\n⚠️  Medium confidence prediction"
        else:
            results += "\n❌ Low confidence prediction"
            
        self.results_text.insert(tk.END, results)
        self.results_text.config(state='disabled')
        
        # Update status
        self.update_status(f"Prediction complete: {self.classes[prediction]}")
        
    def _prediction_error(self, error_msg):
        """Handle prediction errors"""
        self.progress.stop()
        self.predict_btn.config(state='normal', text="Predict")
        self.results_text.config(state='normal')
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"❌ Error during prediction:\n{error_msg}")
        self.results_text.config(state='disabled')
        self.update_status("Prediction failed")
        messagebox.showerror("Prediction Error", f"Failed to predict image: {error_msg}")
        
    def update_status(self, message):
        """Update the status bar"""
        self.status_label.config(text=f"Status: {message}")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = BreastCancerPredictorGUI(root)
    
    # Center the window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()
