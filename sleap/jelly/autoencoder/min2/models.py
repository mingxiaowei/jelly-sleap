import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, LSTM, \
        Dense, Flatten, concatenate, Reshape, TimeDistributed, RepeatVector, \
        MultiHeadAttention, LayerNormalization, Add, Conv2D, Conv1D, Multiply, Conv3D, Embedding
from tensorflow.keras.optimizers import Adam

def masked_mse_loss(y_true, y_pred):
    """
    Compute mean squared error (MSE) only for valid points.
    A point is considered missing if its ground truth is (0, 0).
    """
    # Create a mask: valid points get 1; missing points (0,0) get 0.
    is_missing = tf.logical_and(tf.equal(y_true[..., 0], 0.0),
                                tf.equal(y_true[..., 1], 0.0))
    # is_missing = tf.math.logical_or(
    #     tf.math.is_nan(y_true[..., 0]),
    #     tf.math.is_nan(y_true[..., 1]))
    mask = tf.cast(tf.logical_not(is_missing), tf.float32)  # Shape: (batch, seq_len, 17)
    
    # Compute squared error per coordinate pair.
    print(f'y_true.shape: {y_true.shape}')
    print(f'y_pred.shape: {y_pred.shape}')
    squared_error = tf.square(y_true - y_pred)  # Shape: (batch, seq_len, 17, 2)
    # Sum errors over the two coordinates.
    squared_error = tf.reduce_sum(squared_error, axis=-1)  # Shape: (batch, seq_len, 17)
    
    # Zero-out errors for missing points.
    masked_squared_error = squared_error * mask
    
    # Average only over valid (non-missing) points.
    total_valid = tf.reduce_sum(mask) + K.epsilon()
    mse = tf.reduce_sum(masked_squared_error) / total_valid
    return mse

def get_model_1(window_size=5):
    
    # LSTM autoencoder
    input_seq = Input(shape=(window_size, 34))
    x = LSTM(64, activation='relu', return_sequences=True)(input_seq)
    x = LSTM(32, activation='relu', return_sequences=False)(x)
    x = RepeatVector(window_size)(x)
    x = LSTM(32, activation='relu', return_sequences=True)(x)
    x = LSTM(64, activation='relu', return_sequences=True)(x)
    output = TimeDistributed(Dense(34, activation='sigmoid'))(x)

    model = Model(input_seq, output)
    model.summary()
    
    return model

def get_model_2(window_size=5):
    
    norm_factors = tf.constant([170.0, 174.0])
            
    input_shape = (window_size, 17, 2)
    inputs = layers.Input(shape=input_shape, name='input_sequence')


    norm_inputs = layers.Lambda(lambda x: x / norm_factors, name='normalize')(inputs)
    x = layers.TimeDistributed(layers.Flatten(), name='flatten_per_frame')(norm_inputs)
    x = layers.GaussianNoise(0.05, name='gaussian_noise')(x)

    # --- Encoder ---
    latent = layers.LSTM(64, activation='relu', name='encoder_lstm')(x) # (batch_size, 64)

    # --- Decoder ---
    # Repeat the latent vector for each time step.
    x_decoded = layers.RepeatVector(window_size, name='repeat_vector')(latent)
    # Decode the latent vector into a sequence.
    x_decoded = layers.LSTM(64, activation='relu', return_sequences=True, name='decoder_lstm')(x_decoded)
    # Map back to 34 features using a TimeDistributed Dense layer.
    x_decoded = layers.TimeDistributed(layers.Dense(34), name='time_distributed_dense')(x_decoded)
    # Reshape each time step back to (17, 2).
    outputs = layers.TimeDistributed(layers.Reshape((17, 2)), name='output_sequence')(x_decoded)

    outputs_scaled = layers.Lambda(lambda x: x * norm_factors, name='denormalize')(outputs)

    model = models.Model(inputs, outputs_scaled, name='temporal_autoencoder')
    model.summary()

    return model

def get_model_3(window_size=5, frame_shape=(170, 174, 1), coord_dim=34):
    # multimodel; concatenate video and coordinates
    # Video processing branch
    video_input = Input(shape=(window_size, *frame_shape))
    x = TimeDistributed(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))(video_input)
    x = TimeDistributed(tf.keras.layers.MaxPooling2D((2,2)))(x)
    x = ConvLSTM2D(32, (3,3), activation='relu', return_sequences=True, padding='same')(x)
    x = TimeDistributed(Flatten())(x)
    video_encoder = LSTM(64, return_sequences=False)(x)
    
    # Coordinate processing branch
    coord_input = Input(shape=(window_size, coord_dim))
    y = LSTM(128, return_sequences=True)(coord_input)
    y = LSTM(64, return_sequences=False)(y)
    
    # Combined processing
    combined = concatenate([video_encoder, y])
    
    # Decoder
    z = RepeatVector(window_size)(combined)
    z = LSTM(64, return_sequences=True)(z)
    z = LSTM(128, return_sequences=True)(z)
    coord_output = TimeDistributed(Dense(coord_dim, activation='sigmoid'))(z)
    
    model = Model(inputs=[video_input, coord_input], outputs=coord_output)
    
    return model

def get_model_4(window_size=5, frame_shape=(170, 174, 1), coord_dim=34):
    video_input = Input(shape=(window_size, *frame_shape))
    x = TimeDistributed(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))(video_input)
    x = TimeDistributed(tf.keras.layers.MaxPooling2D((2,2)))(x)
    x = ConvLSTM2D(32, (3,3), activation='tanh', return_sequences=True, padding='same')(x)
    x = TimeDistributed(Flatten())(x)
    video_features = LSTM(128, return_sequences=True)(x)  # Keep temporal dimension
    
    # Coordinate encoder
    coord_input = Input(shape=(window_size, coord_dim))
    y = LSTM(128, return_sequences=True)(coord_input)
    y = LayerNormalization()(y)
    y = LSTM(128, return_sequences=True)(y)
    coord_features = LayerNormalization()(y)
    
    # Multi-head cross attention (coordinates attend to video features)
    attn_output = MultiHeadAttention(
        num_heads=4,
        key_dim=64,
        value_dim=64
    )(query=coord_features, key=video_features, value=video_features)
    
    # Residual connection
    attn_output = Add()([coord_features, attn_output])
    attn_output = LayerNormalization()(attn_output)
    
    # Temporal processing decoder
    z = LSTM(256, return_sequences=True)(attn_output)
    z = LSTM(128, return_sequences=True)(z)
    coord_output = TimeDistributed(Dense(coord_dim, activation='sigmoid'))(z)
    
    model = Model(inputs=[video_input, coord_input], outputs=coord_output)
    model.summary()
    
    return model

def get_model_5(window_size=5, coord_dim=34):
    
    # Input: (batch_size, window_size, 17*2)
    inputs = Input(shape=(window_size, coord_dim))
    
    # Temporal encoder with dilated convolutions
    x = Conv1D(64, 3, dilation_rate=1, padding="causal", activation="relu")(inputs)
    x = LayerNormalization()(x)
    x = Conv1D(128, 3, dilation_rate=2, padding="causal", activation="relu")(x)
    x = LayerNormalization()(x)
    x = Conv1D(256, 3, dilation_rate=4, padding="causal", activation="relu")(x)
    x = LayerNormalization()(x)
    
    # Attention gate for temporal features
    attn = Conv1D(256, 1, activation="sigmoid")(x)
    x = Multiply()([x, attn])
    
    # Bottleneck with residual connection
    encoded = Add()([x, Conv1D(256, 1)(x)])
    encoded = LayerNormalization()(encoded)
    
    # Temporal decoder
    x = Conv1D(128, 3, dilation_rate=2, padding="causal", activation="relu")(encoded)
    x = LayerNormalization()(x)
    x = Conv1D(64, 3, dilation_rate=1, padding="causal", activation="relu")(x)
    x = LayerNormalization()(x)
    
    # Final reconstruction
    decoded = Conv1D(coord_dim, 3, padding="same", activation="sigmoid")(x)
    
    model = Model(inputs, decoded)
    model.summary()
    
    return model

def get_model_6(window_size=5, coord_dim=34, latent_dim=32):
    # simple feedforward autoencoder
    inputs = Input(shape=(window_size, coord_dim))
    x = Dense(128, activation='relu')(inputs)
    x = Dense(64, activation='relu')(x)
    encoded = Dense(latent_dim, activation='relu')(x)

    x = Dense(64, activation='relu')(encoded)
    x = Dense(128, activation='relu')(x)
    decoded = Dense(coord_dim, activation='sigmoid')(x)

    model = Model(inputs, decoded)
    model.summary()
    
    return model

def get_model_7(window_size=5, video_shape=(170, 174, 1), coord_shape=34):
    # Inputs
    video_input = Input(shape=(window_size, *video_shape))  # (B, T, H, W, C)
    coord_input = Input(shape=(window_size, coord_shape))   # (B, T, 34)
    
    # Video processing branch
    x = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(video_input)
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(x)
    video_features = Reshape((window_size, -1))(x)  # (B, T, D)
    
    # Coordinate processing branch
    y = Dense(128, activation='relu')(coord_input)
    y = Dense(256, activation='relu')(y)
    coord_features = LayerNormalization()(y)
    
    # Cross-modal attention
    attention_output = MultiHeadAttention(
        num_heads=4,
        key_dim=64,
        value_dim=64
    )(query=coord_features, key=video_features, value=video_features)
    
    # Feature fusion
    merged = concatenate([coord_features, attention_output], axis=-1)
    merged = Dense(512, activation='relu')(merged)
    merged = LayerNormalization()(merged)
    
    # Temporal decoder
    decoded = Dense(256, activation='relu')(merged)
    decoded = Dense(128, activation='relu')(decoded)
    coord_output = Dense(coord_shape, activation='sigmoid')(decoded)
    
    model = Model(inputs=[video_input, coord_input], outputs=coord_output)
    # model.summary()
    
    return model

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, d_model):
        super().__init__()
        self.pos_encoding = self.positional_encoding(sequence_length, d_model)

    def positional_encoding(self, length, depth):
        depth = depth/2
        positions = np.arange(length)[:, np.newaxis]
        depths = np.arange(depth)[np.newaxis, :]/depth
        angle_rates = 1 / (10000**depths)
        angle_rads = positions * angle_rates
        pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        return x + self.pos_encoding[tf.newaxis, :x.shape[1], :]

def get_model_8(window_size=5, coord_dim=34, num_layers=2):
    inputs = Input(shape=(window_size, coord_dim))
    
    # Positional Encoding
    x = PositionalEncoding(window_size, coord_dim)(inputs)
    
    # Transformer Layers
    for _ in range(num_layers):
        # Self-Attention
        attn = MultiHeadAttention(
            num_heads=4,
            key_dim=64,
            value_dim=64
        )(x, x)
        attn = LayerNormalization()(x + attn)
        
        # Feed Forward
        ffn = Dense(512, activation='relu')(attn)
        ffn = Dense(coord_dim)(ffn)
        x = LayerNormalization()(attn + ffn)
    
    # Final Denoising
    outputs = Dense(coord_dim, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    model.summary()
    
    return model

class LearnablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, window_size, coord_dim):
        super().__init__()
        self.position_emb = Embedding(
            input_dim=window_size, 
            output_dim=coord_dim
        )
        self.window_size = window_size
        
    def call(self, x):
        positions = tf.range(start=0, limit=self.window_size, delta=1)
        positions = tf.expand_dims(positions, axis=0)  # (1, window_size)
        pos_emb = self.position_emb(positions)  # (1, window_size, coord_dim)
        return x + pos_emb

def get_model_9(window_size=5, coord_dim=34, num_layers=2):
    inputs = Input(shape=(window_size, coord_dim))
    
    # 1. Learned Positional Embeddings
    x = LearnablePositionalEncoding(window_size, coord_dim)(inputs)
    
    for _ in range(num_layers):
        # 2. Transformer Encoder Layer
        attn = MultiHeadAttention(
            num_heads=4,
            key_dim=64,
            value_dim=64
        )(x, x)
        x = LayerNormalization()(x + attn)
        
        # 3. Feed-Forward Network
        ffn = Dense(512, activation='relu')(x)
        ffn = Dense(coord_dim)(ffn)
        x = LayerNormalization()(x + ffn)
    
    # 4. Center Frame Prediction
    center_idx = window_size // 2
    outputs = Dense(coord_dim, activation='sigmoid')(x[:, center_idx, :])
    
    model = Model(inputs, outputs)
    model.summary()
    
    return model

class LearnablePositionalEncoding2(tf.keras.layers.Layer):
    def __init__(self, window_size, pt_cnt):
        super().__init__()
        self.position_emb = Embedding(
            input_dim=window_size, 
            output_dim=pt_cnt * 2  # Multiply by 2 for x,y coordinates
        )
        self.window_size = window_size
        self.pt_cnt = pt_cnt
        
    def call(self, x):
        # Reshape input from (batch, window_size, pt_cnt, 2) to (batch, window_size, pt_cnt*2)
        batch_size = tf.shape(x)[0]
        x_reshaped = tf.reshape(x, (batch_size, self.window_size, -1))
        
        positions = tf.range(start=0, limit=self.window_size, delta=1)
        positions = tf.expand_dims(positions, axis=0)  # (1, window_size)
        pos_emb = self.position_emb(positions)  # (1, window_size, pt_cnt*2)
        
        return x_reshaped + pos_emb

def get_model_10(window_size=5, pt_cnt=17, num_layers=2):
    # Input shape now represents (window_size, number_of_points, 2)
    inputs = Input(shape=(window_size, pt_cnt, 2))
    
    # 1. Learned Positional Embeddings
    x = LearnablePositionalEncoding2(window_size, pt_cnt)(inputs)
    
    for _ in range(num_layers):
        # 2. Transformer Encoder Layer
        attn = MultiHeadAttention(
            num_heads=4,
            key_dim=64,
            value_dim=64
        )(x, x)
        x = LayerNormalization()(x + attn)
        
        # 3. Feed-Forward Network
        ffn = Dense(512, activation='relu')(x)
        ffn = Dense(pt_cnt * 2)(ffn)  # Output dimension matches flattened points
        x = LayerNormalization()(x + ffn)
    
    # 4. Center Frame Prediction
    center_idx = window_size // 2
    x = Dense(pt_cnt * 2, activation='sigmoid')(x[:, center_idx, :])
    
    # Reshape output back to (batch, pt_cnt, 2)
    outputs = Reshape((pt_cnt, 2))(x)
    
    model = Model(inputs, outputs)
    model.summary()
    
    return model