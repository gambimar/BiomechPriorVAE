import pickle

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
print(f"Scaler expects {scaler.n_features_in_} features")
print(f"Scaler scale shape: {scaler.scale_.shape}")