import pickle
import os

# Global variables to hold the model and scaler
clf = None
scaler = None
pca = None
def load_model():
    global clf, scaler, pca
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, '../models/naive_bayes_model.pkl')
    scaler_path = os.path.join(base_dir, '../models/scaler.pkl')
    PCA_path = os.path.join(base_dir, '../models/pca_models.pkl')

    try:
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                clf = pickle.load(f)
            print("Model loaded successfully.")
        else:
            print(f"Model file not found at {model_path}")

        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("Scaler loaded successfully.")
        else:
            print(f"Scaler file not found at {scaler_path}")
        
        if os.path.exists(PCA_path):
            with open(PCA_path, 'rb') as f:
                pca = pickle.load(f)
            print("PCA loaded successfully.")
        else:
            print(f"PCA file not found at {PCA_path}")
    except Exception as e:
        print(f"Error loading models: {e}")

# Load models on import
load_model()
