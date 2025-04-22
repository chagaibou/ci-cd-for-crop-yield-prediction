import joblib

# Sauvegarder le modèle

def save_model(model,model_name='model.plk'):
    joblib.dump(model,model_name)