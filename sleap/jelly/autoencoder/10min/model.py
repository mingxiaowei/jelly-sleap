import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

def multihead_attention_model():
    # Inputs
    video_input = layers.Input(shape=(170, 174, 1), name="video_input")
    points_input = layers.Input(shape=(17, 2), name="points_input")
    
    # Video encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(video_input)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    video_features = x
    video_tokens = layers.Reshape((-1, 64))(video_features)
    
    # Point encoder
    point_queries = layers.Dense(64, activation="relu")(points_input)
    
    # Cross-attention (GPU-optimized implementation)
    attention = layers.MultiHeadAttention(num_heads=4, key_dim=16)(point_queries, video_tokens)
    x = layers.Concatenate(axis=-1)([attention, point_queries])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    corrected_points = layers.Dense(2, name="corrected_points")(x)
    
    model = tf.keras.Model(
        inputs=[video_input, points_input],
        outputs=corrected_points
    )
    return model
    
def feedforward_model_pts_only():
    points_input = layers.Input(shape=(17, 2), name="points_input")
    
    # Encoder
    x = layers.Flatten()(points_input)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    encoded = layers.Dense(16, activation='relu', name='encoded')(x)
    
    # Decoder
    x = layers.Dense(32, activation='relu')(encoded)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(34, activation='relu')(x)  # 17 points * 2 coordinates
    decoded = layers.Reshape((17, 2), name='decoded')(x)
    
    model = tf.keras.Model(
        inputs=points_input,
        outputs=decoded,
        name='points_autoencoder'
    )
    return model

non_missing_frame_indices_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/non_missing_frame_indices.npy'

def labeled_loss_from_indices(indices_path=non_missing_frame_indices_path):
    indices = np.load(indices_path)
    indices_tensor = tf.constant(indices, dtype=tf.int32)
    def labeled_loss(y_true, y_pred):
        y_true = tf.gather(y_true, indices_tensor, axis=0)
        y_pred = tf.gather(y_pred, indices_tensor, axis=0)
        return tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    return labeled_loss

def temporal_pts_only_model():
    pass