import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import Model

def masked_mse_loss(y_true, y_pred):
    """
    Compute mean squared error (MSE) only for valid points.
    A point is considered missing if its ground truth is (0, 0).
    """
    # Create a mask: valid points get 1; missing points (0,0) get 0.
    is_missing = tf.logical_and(tf.equal(y_true[..., 0], 0.0),
                                tf.equal(y_true[..., 1], 0.0))
    mask = tf.cast(tf.logical_not(is_missing), tf.float32)  # Shape: (batch, seq_len, 17)
    
    # Compute squared error per coordinate pair.
    squared_error = tf.square(y_true - y_pred)  # Shape: (batch, seq_len, 17, 2)
    # Sum errors over the two coordinates.
    squared_error = tf.reduce_sum(squared_error, axis=-1)  # Shape: (batch, seq_len, 17)
    
    # Zero-out errors for missing points.
    masked_squared_error = squared_error * mask
    
    # Average only over valid (non-missing) points.
    total_valid = tf.reduce_sum(mask) + K.epsilon()
    mse = tf.reduce_sum(masked_squared_error) / total_valid
    return mse

def split_train_val(X, train_size=0.8):
    train_size = int(train_size * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    return X_train, X_val

def load_data_1(slice_idx=2500, window_size=5):
    coords = np.load('/home/mingxiao/Desktop/jellyfish/video/video_1_clips/manual_5min_c0_points.npy')
    coords_sliced = coords[:slice_idx].copy()
    X = get_sliding_windows_1(coords, window_size=window_size, flat_len=9000)
    X_sliced = get_sliding_windows_1(coords_sliced, window_size=window_size, flat_len=slice_idx)
    return X, X_sliced

def get_sliding_windows_1(coords, window_size=5, flat_len=2500):
    coords_normalized = coords.astype('float32') / [170, 174]
    coords_flat = coords_normalized.reshape((flat_len, -1))
    X = []
    first_half_window = window_size // 2
    second_half_window = window_size - first_half_window
    for i in range(coords_flat.shape[0]):
        start = max(0, i - first_half_window)
        end = min(coords_flat.shape[0], i + second_half_window)
        pad_before = max(0, first_half_window - i)
        pad_after = max(0, (i + second_half_window) - coords_flat.shape[0])
    
        window = coords_flat[start:end]
        if pad_before > 0 or pad_after > 0:
            window = np.pad(window, ((pad_before, pad_after), (0, 0)), mode='constant')
        X.append(window)

    return np.array(X)

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
    model.compile(optimizer='adam', loss=masked_mse_loss)
    model.summary()
    
    return model

def run_model_1(train_slice_only=True, window_size=5):
    
    X, X_sliced = load_data_1(window_size=window_size)
    if train_slice_only:
        X_train, X_val = split_train_val(X_sliced)
    else:
        X_train, X_val = split_train_val(X)
    
    model = get_model_1(window_size=window_size)
    model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val))

    denoised_windows = model.predict(X)
    denoised_coords = denoised_windows[:, window_size // 2, :]  # Extract middle frame

    denoised_coords = denoised_coords.reshape((-1, 17, 2))
    denoised_coords = denoised_coords * [170, 174]  # Scale back to original coordinates
    
    return denoised_coords

def load_data_2(slice_idx=2500, window_size=5):
    coords = np.load('/home/mingxiao/Desktop/jellyfish/video/video_1_clips/manual_5min_c0_points.npy')
    coords_sliced = coords[:slice_idx].copy()
    X = get_sliding_windows_2(coords, window_size=window_size)
    X_sliced = get_sliding_windows_2(coords_sliced, window_size=window_size)
    return X, X_sliced

def get_sliding_windows_2(coords, window_size=5):
    X = []
    for i in range(len(coords) - window_size + 1):
        X.append(coords[i:i+window_size])
    return np.array(X)

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
    model.compile(optimizer='adam', loss=masked_mse_loss)
    model.summary()

    return model

def reconstruct_full_sequence(predicted_seqs, num_frames=9000):
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

def run_model_2(train_slice_only=True, window_size=5):
    
    X, X_sliced = load_data_2(window_size=window_size)
    if train_slice_only:
        X_train, X_val = split_train_val(X_sliced)
    else:
        X_train, X_val = split_train_val(X)
    
    model = get_model_2(window_size=window_size)
    model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_val, X_val))

    denoised_windows = model.predict(X)
    denoised_coords = reconstruct_full_sequence(denoised_windows)
    
    return denoised_coords