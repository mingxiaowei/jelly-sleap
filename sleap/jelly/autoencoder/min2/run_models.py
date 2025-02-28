import numpy as np
import tensorflow as tf
import datetime
from .models import *

import sys
sys.path.append('/home/mingxiao/Desktop/jelly-sleap/sleap/jelly/autoencoder/src')
from src.data_loader import load_data
from src.model_classes import *

def model_runner(model, *data, 
                 epochs=50, 
                 batch_size=32, 
                 optimizer='adam',
                 loss=masked_mse_loss,
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
                 save_path=None,
                 save_affix=None):
    
    model.compile(optimizer=optimizer, loss=loss)
    model.summary()
    X_train, X_val, y_train, y_val, X = data
    
    model.fit(X_train, y_train, 
              epochs=epochs, 
              batch_size=batch_size, 
              validation_data=(X_val, y_val),
              callbacks=callbacks)
    
    predictions = model.predict(X)
    predictions = model.postprocess(predictions)
    return predictions

def run_model_1(train_slice_only=True, window_size=5, epochs=50, batch_size=32, optimizer='adam'):
    
    X_train, X_val, y_train, y_val, X = load_data(window_size=window_size, augment=True, load_video=False)
        
    model = Model1(window_size=window_size)
    
    return model_runner(model, X_train, X_val, y_train, y_val, X, epochs=epochs, batch_size=batch_size, optimizer=optimizer)

def load_data_2(slice_idx=2500, window_size=5):
    return load_data(get_sliding_windows_2, slice_idx, window_size)

def get_sliding_windows_2(coords, window_size=5, flat_len=None):
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

def run_model_2(train_slice_only=True, window_size=5, epochs=50, batch_size=32, optimizer='adam'):
    
    X_train, X_val, y_train, y_val, X = load_data(window_size=window_size, augment=True, load_video=False, flatten=False)
    
    model = get_model_2(window_size=window_size)
    model.compile(optimizer=optimizer, loss=masked_mse_loss)
    model.summary()
    model.fit(X_train, y_train, 
              epochs=epochs, 
              batch_size=batch_size, 
              validation_data=(X_val, y_val),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

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
        window = np.pad(window, ((pad_before, pad_after), *[(0,0)]*(len(data.shape)-1)), mode='edge')
        windows.append(window)
    return np.array(windows)


def load_data_3(slice_idx=2500, window_size=5):
    
    coords_corrected = np.load('/home/mingxiao/Desktop/jellyfish/video/video_1_clips/manual_5min_c0_points.npy')
    coords_raw = np.load('/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_raw_points.npy')
    video_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_high_res_5min_track_reencoded_0.mp4'
    video = load_video(video_path)
    
    frames_normalized = np.expand_dims(video, -1)  # Add channel dimension

    coords_raw_normalized = coords_raw.astype('float32') / [170, 174]
    coords_raw_flat = coords_raw_normalized.reshape((9000, 34))  # Flatten coordinates
    coords_corrected_normalized = coords_corrected.astype('float32') / [170, 174]
    coords_corrected_flat = coords_corrected_normalized.reshape((9000, 34))  # Flatten coordinates

    frames_windows = get_sliding_windows_3(frames_normalized, window_size=window_size)
    coords_raw_windows = get_sliding_windows_3(coords_raw_flat, window_size=window_size)
    coords_corrected_windows = get_sliding_windows_3(coords_corrected_flat, window_size=window_size)
    
    coords_raw_sliced = coords_raw_flat[:slice_idx]
    coords_corrected_sliced = coords_corrected_flat[:slice_idx]
    frames_sliced = frames_normalized[:slice_idx]
    
    coords_raw_windows_sliced = get_sliding_windows_3(coords_raw_sliced, window_size=window_size)
    coords_corrected_windows_sliced = get_sliding_windows_3(coords_corrected_sliced, window_size=window_size)
    frames_windows_sliced = get_sliding_windows_3(frames_sliced, window_size=window_size)
    
    return coords_raw_windows, coords_corrected_windows, frames_windows, \
        coords_raw_windows_sliced, coords_corrected_windows_sliced, frames_windows_sliced

def run_model_3(train_slice_only=True, window_size=5, epochs=50, batch_size=32, optimizer='adam'):  
    X_train, X_val, y_train, y_val, X = load_data(window_size=window_size, augment=True, load_video=True)
    
    model = get_model_3(window_size=window_size)
    model.compile(optimizer=optimizer, loss=masked_mse_loss)
    model.summary()

    # Train the model
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    )
    
    denoised_coords = model.predict(X)
    
    denoised_coords = denoised_coords.reshape((-1, 17, 2))
    denoised_coords = denoised_coords * [170, 174]  # Scale back to original coordinates
    denoised_coords = denoised_coords[window_size//2::window_size]
    
    return denoised_coords

def run_model_4(train_slice_only=True, window_size=5, epochs=100, batch_size=32, optimizer='adam'):
    X_train, X_val, y_train, y_val, X = load_data(window_size=window_size, augment=True, load_video=True)
        
    model = get_model_4(window_size=window_size)
    model.compile(optimizer=optimizer, loss=masked_mse_loss)
    model.summary()
    
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
        ]
    )

    # Generate improved coordinates
    denoised_windows = model.predict(X)
    denoised_coords = denoised_windows[:, window_size//2, :].reshape(-1, 17, 2) * [170, 174]
    
    return denoised_coords

def load_data_5(slice_idx=2500, window_size=5):
    coords_corrected = np.load('/home/mingxiao/Desktop/jellyfish/video/video_1_clips/manual_5min_c0_points.npy')
    coords_raw = np.load('/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_raw_points.npy')
    
    coords_corrected_normalized = (coords_corrected / np.array([170, 174])).reshape((9000, 34))
    coords_raw_normalized = (coords_raw / np.array([170, 174])).reshape((9000, 34))
    
    coords_corrected_windows = get_sliding_windows_3(coords_corrected_normalized, window_size=window_size)
    coords_raw_windows = get_sliding_windows_3(coords_raw_normalized, window_size=window_size)
    
    coords_corrected_sliced = coords_corrected_normalized[:slice_idx]
    coords_raw_sliced = coords_raw_normalized[:slice_idx]
    
    coords_corrected_windows_sliced = get_sliding_windows_3(coords_corrected_sliced, window_size=window_size)
    coords_raw_windows_sliced = get_sliding_windows_3(coords_raw_sliced, window_size=window_size)
    
    return coords_raw_windows, coords_raw_windows_sliced, \
        coords_corrected_windows, coords_corrected_windows_sliced
        
    
def run_model_5(train_slice_only=True, window_size=5, epochs=100, batch_size=32, optimizer='adam'):
    
    X_train, X_val, y_train, y_val, X = load_data(window_size=window_size, augment=True, load_video=False)
        
    model = get_model_5(window_size=window_size)
    model.compile(optimizer=optimizer, loss=masked_mse_loss)
    model.summary()
    
    model.fit(X_train, y_train, 
              epochs=epochs, 
              batch_size=batch_size, 
              validation_data=(X_val, y_val),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    
    denoised = model.predict(X)
    improved_coords = denoised[:, window_size//2, :]  # Extract center frame
    improved_coords = improved_coords.reshape(-1, 17, 2) * np.array([170, 174])
    
    return improved_coords

def run_model_6(train_slice_only=True, window_size=5, epochs=100, batch_size=32, optimizer='adam'):
    
    X_train, X_val, y_train, y_val, X = load_data(window_size=window_size, augment=True, load_video=False)
    
    model = get_model_6(window_size=window_size)
    model.compile(optimizer=optimizer, loss=masked_mse_loss)
    model.summary()
    
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    )
    
    denoised = model.predict(X)
    improved_coords = denoised[:, window_size//2, :]  # Extract center frame
    improved_coords = improved_coords.reshape(-1, 17, 2) * np.array([170, 174])
    
    return improved_coords

def run_model_7(train_slice_only=True, window_size=5, epochs=100, batch_size=32, optimizer='adam', model_save_path=None):
    # Force CPU usage
    with tf.device('/CPU:0'):
        X_train, X_val, y_train, y_val, X = load_data(window_size=window_size, augment=True, load_video=True)   
        
        model = get_model_7(window_size=window_size)
        model.compile(optimizer=optimizer, loss=masked_mse_loss)
        model.summary()
        
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1)]
        )
        if model_save_path is not None:
            model.save(model_save_path)
        
        denoised = model.predict(X)
        denoised_coords = denoised[:, window_size//2, :]  # Extract center frame
        denoised_coords = denoised_coords.reshape(-1, 17, 2) * np.array([170, 174])
        
        return denoised_coords
    
def run_model_8(window_size=5, epochs=100, batch_size=32, optimizer='adam', split_size=0.85, num_layers=2, shuffle=True):
    X_train, X_val, y_train, y_val, X = load_data(window_size=window_size, augment=True, load_video=True)
    
    model = get_model_8(window_size=window_size, num_layers=num_layers)
    model.compile(optimizer=optimizer, loss=masked_mse_loss)
    model.summary()
    
    model.fit(X_train, y_train, 
              epochs=epochs, 
              batch_size=batch_size, 
              validation_data=(X_val, y_val),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
    
    denoised_windows = model.predict(X)
    improved_coords = denoised_windows[:, window_size//2, :].reshape(-1, 17, 2) * np.array([170, 174])
    
    return improved_coords

def run_model_9(window_size=5, epochs=100, batch_size=32, 
                optimizer='adam', split_size=0.9, 
                num_layers=2, 
                shuffle=True,
                dropout_rate=0.01, swap_rate=0.005):
    X_train, X_val, y_train, y_val, X = load_data(
        window_size=window_size, 
        augment=True, 
        load_video=False, 
        split_size=split_size, 
        shuffle=shuffle, 
        dropout_rate=dropout_rate, 
        swap_rate=swap_rate,
    )
    
    model = get_model_9(window_size=window_size, num_layers=num_layers)
    model.compile(optimizer=optimizer, loss=masked_mse_loss)
    model.summary()
    
    model.fit(X_train, y_train[:, window_size//2, :], 
              epochs=epochs, batch_size=batch_size, 
              validation_data=(X_val, y_val[:, window_size//2, :]),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=1)])
    
    denoised_coords = model.predict(X)
    denoised_coords = denoised_coords.reshape((-1, 17, 2)) * np.array([170, 174])
    return denoised_coords

def run_model_10(window_size=5, epochs=100, batch_size=32, 
                optimizer='adam', split_size=0.9, 
                num_layers=2, 
                shuffle=True,
                dropout_rate=0.01, 
                swap_rate=0.005, 
                roll=True):
    
    X_train, X_val, y_train, y_val, X = load_data(
        window_size=window_size, 
        augment=True, 
        load_video=False, 
        split_size=split_size, 
        shuffle=shuffle, 
        dropout_rate=dropout_rate, 
        swap_rate=swap_rate,
        roll=roll,
        flatten=False,
    )
    
    model = get_model_10(window_size=window_size, num_layers=num_layers)
    model.compile(optimizer=optimizer, loss=masked_mse_loss)
    model.summary()
    
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    model.fit(X_train, y_train, 
              epochs=epochs, batch_size=batch_size, 
              validation_data=(X_val, y_val),
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=1), tensorboard_callback])
    
    denoised_coords = model.predict(X)
    denoised_coords = denoised_coords.reshape((-1, 17, 2))
    # * np.array([170, 174])
    return denoised_coords