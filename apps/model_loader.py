
# app/model_loader.py
import pickle

def load_model(model_name):
    model_path = f'C:/Users/adnan/AI-Medical-Diagnosis-System/models/{model_name}'
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Load models
diabetes_model = load_model('diabetes_model.pkl')
lung_cancer_model = load_model('lung_cancer_model.pkl')
