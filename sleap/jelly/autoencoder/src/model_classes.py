import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, ConvLSTM2D, LSTM, \
        Dense, Flatten, concatenate, Reshape, TimeDistributed, RepeatVector, \
        MultiHeadAttention, LayerNormalization, Add, Conv2D, Conv1D, Multiply, Conv3D, Embedding
from data_loader import load_data

class BaseModel:
    
    canvas_size = (170, 174)
    pts_cnt = 17
    model = None
    
    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)  
    
    def postprocess(self, predicted_seqs):
        return predicted_seqs
    
    def summary(self):
        self.model.summary()
    
    def load_data(self):
        raise NotImplementedError("Subclasses must implement this method")
    

class Model1(BaseModel):
    
    def __init__(self, window_size=5):
        input_seq = Input(shape=(window_size, 34))
        x = LSTM(64, activation='relu', return_sequences=True)(input_seq)
        x = LSTM(32, activation='relu', return_sequences=False)(x)
        x = RepeatVector(window_size)(x)
        x = LSTM(32, activation='relu', return_sequences=True)(x)
        x = LSTM(64, activation='relu', return_sequences=True)(x)
        output = TimeDistributed(Dense(34, activation='sigmoid'))(x)
        
        self.model = Model(input_seq, output)
        self.window_size = window_size
    
    def postprocess(self, predicted_seqs):
        denoised_coords = predicted_seqs[:, self.window_size // 2, :]  # Extract middle frame
        denoised_coords = denoised_coords.reshape((-1, self.pts_cnt, 2))
        denoised_coords = denoised_coords * self.canvas_size  # Scale back to original coordinates
        return denoised_coords

class Model2(BaseModel):
    
    def __init__(self, window_size=5):
        
        input_shape = (window_size, 17 * 2)
        inputs = layers.Input(shape=input_shape, name='input_sequence')

        norm_inputs = layers.Lambda(lambda x: x / self.canvas_size, name='normalize')(inputs)
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

        outputs_scaled = layers.Lambda(lambda x: x * self.canvas_size, name='denormalize')(outputs)

        model = models.Model(inputs, outputs_scaled, name='temporal_autoencoder')
        self.model = model
        self.window_size = window_size
    
    def postprocess(self, predicted_seqs, num_frames=9000):
        # Initialize an array to accumulate predictions and an array to count contributions.
        output = np.zeros((num_frames, 17, 2), dtype=np.float32)
        counts = np.zeros((num_frames, 1), dtype=np.float32)
        
        num_seqs = predicted_seqs.shape[0]
        seq_length = predicted_seqs.shape[1]
        for i in range(num_seqs):
            for j in range(seq_length):
                frame_idx = i + j
                if frame_idx < num_frames:
                    output[frame_idx] += predicted_seqs[i, j]
                    counts[frame_idx] += 1
        # Avoid division by zero and compute the average.
        output /= np.maximum(counts[:,:,None], 1)
        return output

class Model3(BaseModel):
    
    def __init__(self, window_size=5):
        # multimodel; concatenate video and coordinates
        # Video processing branch
        frame_shape = self.canvas_size + (1,)
        coord_dim = self.pts_cnt * 2
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
        
        self.model = Model(inputs=[video_input, coord_input], outputs=coord_output)
        self.window_size = window_size
    
    def postprocess(self, predicted_seqs):
        denoised_coords = predicted_seqs.reshape((-1, 17, 2))
        denoised_coords = denoised_coords * self.canvas_size  # Scale back to original coordinates
        denoised_coords = denoised_coords[self.window_size//2::self.window_size]
        return denoised_coords

class Model4(BaseModel):
    
    def __init__(self, window_size=5):
        frame_shape = self.canvas_size + (1,)
        coord_dim = self.pts_cnt * 2
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
        
        self.model = Model(inputs=[video_input, coord_input], outputs=coord_output)
        self.window_size = window_size
    
    def postprocess(self, predicted_seqs):
        denoised_coords = predicted_seqs[:, self.window_size//2, :].reshape(-1, 17, 2) * self.canvas_size
        return denoised_coords

class Model5(BaseModel):
    
    def __init__(self, window_size=5):
        
        coord_dim = self.pts_cnt * 2
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
        
        self.model = Model(inputs, decoded)
        self.window_size = window_size
        
    def postprocess(self, predicted_seqs):
        improved_coords = predicted_seqs[:, self.window_size//2, :]  # Extract center frame
        improved_coords = improved_coords.reshape(-1, 17, 2) * self.canvas_size
        return improved_coords

class Model6(BaseModel):
    
    def __init__(self, window_size=5, latent_dim=32):
        # simple feedforward autoencoder
        coord_dim = self.pts_cnt * 2
        inputs = Input(shape=(window_size, coord_dim))
        x = Dense(128, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        encoded = Dense(latent_dim, activation='relu')(x)

        x = Dense(64, activation='relu')(encoded)
        x = Dense(128, activation='relu')(x)
        decoded = Dense(coord_dim, activation='sigmoid')(x)

        self.model = Model(inputs, decoded)
        self.window_size = window_size
    
    def postprocess(self, predicted_seqs):
        improved_coords = predicted_seqs[:, self.window_size//2, :]  # Extract center frame
        improved_coords = improved_coords.reshape(-1, 17, 2) * self.canvas_size
        return improved_coords

class Model7(BaseModel):
    
    def __init__(self, window_size=5):
        video_shape = self.canvas_size + (1,)
        coord_shape = self.pts_cnt * 2
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
        
        self.model = Model(inputs=[video_input, coord_input], outputs=coord_output)
        self.window_size = window_size
    
    def postprocess(self, predicted_seqs):
        denoised_coords = predicted_seqs[:, self.window_size//2, :].reshape(-1, 17, 2) * self.canvas_size
        return denoised_coords

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

class Model8(BaseModel):
    
    def __init__(self, window_size=5, num_layers=2):
        coord_dim = self.pts_cnt * 2
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
        
        self.model = Model(inputs, outputs)
        self.window_size = window_size
    
    def postprocess(self, predicted_seqs):
        denoised_coords = predicted_seqs[:, self.window_size//2, :].reshape(-1, 17, 2) * self.canvas_size
        return denoised_coords

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

class Model9(BaseModel):
    
    def __init__(self, window_size=5, num_layers=2):
        coord_dim = self.pts_cnt * 2
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
        
        self.model = Model(inputs, outputs)
        self.window_size = window_size
    
    def postprocess(self, predicted_seqs):
        denoised_coords = predicted_seqs.reshape(-1, 17, 2) * self.canvas_size
        return denoised_coords
