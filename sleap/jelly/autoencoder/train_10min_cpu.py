from model import *
from load_data_10min import *
import os

if __name__ == "__main__":
    model = build_gpu_model()
    model.summary()
    train_data, val_data = load_data_cpu()
    
    model.compile(optimizer="adam", loss="mse")
    
    history = model.fit(
        train_data,
        epochs=20,
        validation_data=val_data,  # Use explicit validation dataset
        verbose=1
    )
    
    model_save_path = f'/home/mingxiao/Desktop/jelly-sleap/sleap/jelly/autoencoder/models/10min_cpu_model.keras'
    model_parent_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_parent_dir):
        os.makedirs(model_parent_dir)
    model.save(model_save_path)