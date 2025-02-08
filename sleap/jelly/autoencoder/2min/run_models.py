import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from models import *
from load_data_2min import load_video

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

def get_sliding_windows_3(data, window_size=5):
    windows = []
    first_half_window = window_size // 2
    second_half_window = window_size - first_half_window
    for i in range(len(data)):
        start = max(0, i - first_half_window)
        end = min(len(data), i + second_half_window)
        pad_before = max(0, first_half_window - i)
        pad_after = max(0, (i + second_half_window) - len(data))
        window = data[start:end]
        window = np.pad(window, ((pad_before, pad_after), *[(0,0)]*(len(data.shape)-1)), mode='constant')
        windows.append(window)
    return np.array(windows)


def load_data_3(slice_idx=2500, window_size=5):
    
    coords = np.load('/home/mingxiao/Desktop/jellyfish/video/video_1_clips/manual_5min_c0_points.npy')
    video_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_high_res_5min_track_reencoded_0.mp4'
    video = load_video(video_path)
    
    frames_normalized = np.expand_dims(video, -1)  # Add channel dimension

    coords_normalized = coords.astype('float32') / [170, 174]
    coords_flat = coords_normalized.reshape((9000, 34))  # Flatten coordinates
    
    coords_windows = get_sliding_windows_3(coords_flat, window_size=window_size)
    frames_windows = get_sliding_windows_3(frames_normalized, window_size=window_size)
    
    coords_sliced = coords_flat[:slice_idx]
    frames_sliced = frames_normalized[:slice_idx]
    coords_windows_sliced = get_sliding_windows_3(coords_sliced, window_size=window_size)
    frames_windows_sliced = get_sliding_windows_3(frames_sliced, window_size=window_size)
    
    return coords_windows, frames_windows, coords_windows_sliced, frames_windows_sliced

def run_model_3(train_slice_only=True, window_size=5):  
    coords_windows, frames_windows, coords_windows_sliced, frames_windows_sliced = load_data_3(window_size=window_size)
    if train_slice_only:
        coords_train, coords_val = split_train_val(coords_windows_sliced)
        frames_train, frames_val = split_train_val(frames_windows_sliced)
    else:
        coords_train, coords_val = split_train_val(coords_windows)
        frames_train, frames_val = split_train_val(frames_windows)
    
    model = get_model_3(window_size=window_size)
    model.compile(optimizer=Adam(0.001), loss=masked_mse_loss)
    model.summary()

    # Train the model
    model.fit(
        [frames_train, coords_train], coords_train,
        validation_data=([frames_val, coords_val], coords_val),
        epochs=50,
        batch_size=32
    )
    
    denoised_coords = model.predict([frames_windows, coords_windows])
    
    denoised_coords = denoised_coords.reshape((-1, 17, 2))
    denoised_coords = denoised_coords * [170, 174]  # Scale back to original coordinates
    
    return denoised_coords

def run_model_4(train_slice_only=True, window_size=5):
    coords_windows, frames_windows, coords_windows_sliced, frames_windows_sliced = load_data_3(window_size=window_size)
    if train_slice_only:
        coords_train, coords_val = split_train_val(coords_windows_sliced)
        frames_train, frames_val = split_train_val(frames_windows_sliced)
    else:
        coords_train, coords_val = split_train_val(coords_windows)
        frames_train, frames_val = split_train_val(frames_windows)
        
    model = get_model_4(window_size=window_size)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss=masked_mse_loss)
    model.summary()
    
    model.fit(
        [frames_train, coords_train], coords_train,
        validation_data=([frames_val, coords_val], coords_val),
        epochs=100,
        batch_size=32,
        callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
    )

    # Generate improved coordinates
    denoised_windows = model.predict([frames_windows, coords_windows])
    denoised_coords = denoised_windows[:, window_size//2, :].reshape(-1, 17, 2) * [170, 174]
    
    return denoised_coords