import tensorflow as tf
from tensorflow.keras import layers

def build_model():
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