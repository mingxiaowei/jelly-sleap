from model import *
from load_data_10min import *
import os

if __name__ == "__main__":
    model = multihead_attention_model()
    model.summary()
    train_data, val_data = load_data_cpu()
    
    non_missing_frame_indices = np.load('/home/mingxiao/Desktop/jellyfish/video/video_1_clips/non_missing_frame_indices.npy')
    non_missing_frame_indices_tensor = tf.constant(non_missing_frame_indices, dtype=tf.int32)

    def labeled_loss(y_true, y_pred):
        y_true = tf.gather(y_true, non_missing_frame_indices_tensor, axis=0)
        y_pred = tf.gather(y_pred, non_missing_frame_indices_tensor, axis=0)
        return tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    
    # model.compile(optimizer="adam", loss=labeled_loss)
    model.compile(optimizer="adam", loss="mse")
    
    history = model.fit(
        train_data,
        epochs=20,
        validation_data=val_data,  # Use explicit validation dataset
        verbose=1
    )
    
    model_save_path = f'/home/mingxiao/Desktop/jelly-sleap/sleap/jelly/autoencoder/models/10min_cpu_model_v4.keras'
    model_parent_dir = os.path.dirname(model_save_path)
    if not os.path.exists(model_parent_dir):
        os.makedirs(model_parent_dir)
    model.save(model_save_path)