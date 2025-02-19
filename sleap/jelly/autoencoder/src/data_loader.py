import numpy as np
import cv2
import tensorflow as tf
import sys

sys.path.append('../..')
from python.animation import *
from python.postprocess import *
from python.polygon_based_correction import *

video_path = "/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_high_res_5min_track_reencoded_0.mp4"
corrected_points_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/manual_5min_c0_tracked_points.npy'
raw_points_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_raw_points.npy'

def load_points(raw_points_path=raw_points_path, corrected_points_path=corrected_points_path):
    coords_corrected = np.load(corrected_points_path).astype('float32')
    coords_raw = np.load(raw_points_path).astype('float32')
    
    # Replace (0,0) coordinates with (nan, nan)
    # coords_corrected[(coords_corrected == 0).all(axis=2)] = np.nan
    # coords_raw[(coords_raw == 0).all(axis=2)] = np.nan
    
    return coords_raw, coords_corrected

def video_loader(video_path=video_path, target_size=None, load_as_tensor=False):
    """
    Load an MP4 video into a numpy array (grayscale).
    
    Args:
        video_path: Path to the MP4 file.
        target_size: (width, height) to resize frames (optional).
    
    Returns:
        video_array: Shape (num_frames, height, width)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if target_size:
            gray = cv2.resize(gray, target_size)  # (width, height)
        frames.append(gray)
    
    cap.release()
    video_array = np.array(frames)  # Shape: (num_frames, height, width)
    
    if load_as_tensor:
        video_tensor = tf.constant(video_array, dtype=tf.float16) / 255.0
        video_tensor = tf.expand_dims(video_tensor, axis=-1)  # Add channel dim
        return video_tensor
    else:
        return video_array / 255.0
    
def get_sliding_windows(data, window_size=5):
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

def get_windows_wrapper(coords_list, window_size=5, normalizer=(170, 174), flatten=True):
    coords_windows = []
    for coords in coords_list:
        coords_normalized = coords / normalizer
        if flatten:
            coords_flat = coords_normalized.reshape((len(coords), -1))
            coords_windows.append(get_sliding_windows(coords_flat, window_size=window_size))
        else:
            coords_windows.append(get_sliding_windows(coords_normalized, window_size=window_size))
    return coords_windows

def get_dropout_mask(frame_cnt, window_size, pt_cnt, dropout_rate=0.01):
    mask = np.zeros((frame_cnt, pt_cnt))
    total_elements = frame_cnt * pt_cnt
    num_ones = int(total_elements * dropout_rate)
    indices = np.random.choice(total_elements, num_ones, replace=False)
    rows = indices // pt_cnt
    cols = indices % pt_cnt
    mask[rows, cols] = 1
    mask = np.tile(mask[:, np.newaxis, :], (1, window_size, 1))
    return mask

def augment_points(coords, dropout_rate=0.01, swap_rate=0.005):
    frame_cnt, window_size, pt_cnt = coords.shape[:3]
    
    # 1. dropout
    dropout_mask = get_dropout_mask(frame_cnt, window_size, pt_cnt, dropout_rate)
    # coords[dropout_mask == 1] = np.nan
    coords[dropout_mask == 1] = 0
    
    # 2. swap
    swap_frames = np.random.choice(frame_cnt, int(frame_cnt * swap_rate), replace=False)
    for frame_idx in swap_frames:
        swap_pts = np.random.choice(pt_cnt, np.random.randint(2, 5), replace=False)
        swap_pts_shuffled = np.random.permutation(swap_pts)
        coords[frame_idx, :, swap_pts] = coords[frame_idx, :, swap_pts_shuffled]
        
    return coords

def split_train_val(*data, train_size=0.8):
    train_size = int(train_size * len(data[0]))
    splits = []
    for d in data:
        assert len(d) == len(data[0]), "All data must have the same length"
        splits.extend([d[:train_size], d[train_size:]])
    return splits

def load_data(
    raw_points_path=raw_points_path, 
    corrected_points_path=corrected_points_path, 
    video_path=video_path, 
    load_video=False,
    window_size=30,
    load_as_tensor=False,
    flatten=False,
    shuffle=True,
    augment=False, 
    dropout_rate=0.01,
    swap_rate=0.01, 
    split_size=0.9,
    ):
    _, coords_corrected = load_points(raw_points_path=raw_points_path, 
                                               corrected_points_path=corrected_points_path)
    coords_corrected_copy = coords_corrected.copy()
    coords_corrected = mean_interpolate(coords_corrected)
    # 1. get window
    coords_window, coords_window_copy = get_windows_wrapper([coords_corrected, coords_corrected_copy], window_size=window_size, flatten=flatten)
    X = coords_window_copy # for prediction
    
    # 2. reorder by polygon
    missing_mask = get_missing_mask(coords_corrected)
    coords_window = reorder_by_polygon(coords_window, missing_mask)
    
    # 3. augment
    if augment:
        coords_window_augmented = augment_points(coords_window, dropout_rate, swap_rate)
    else:
        coords_window_augmented = coords_window
        
    # 4. shuffle among windows 
    frame_indices = np.arange(len(coords_window_augmented))
    if shuffle:
        np.random.shuffle(frame_indices)
    coords_window_augmented = coords_window_augmented[frame_indices]
    coords_corrected = coords_corrected[frame_indices]
    
    # 5. train val split
    X_train, X_val, y_train, y_val = split_train_val(coords_window_augmented, coords_corrected, 
                                                     train_size=split_size)
    
    if load_video:
        video = video_loader(video_path=video_path, target_size=None, load_as_tensor=load_as_tensor)
        video_windows = get_sliding_windows(video, window_size=window_size)
        video_original = video_windows
        video_windows = video_windows[indices]
        video_train, video_val = split_train_val(video_windows, train_size=split_size)
        X_train = [video_train, X_train]
        X_val = [video_val, X_val]
        X = [video_original, X]
        
    return X_train, X_val, y_train, y_val, X

def mean_interpolate(coords_augmented, coords_corrected=None):
    if coords_corrected is None:
        coords_corrected = coords_augmented
    non_missing_indices = np.where(coords_augmented != 0)
    non_missing_coords = coords_corrected[non_missing_indices]
    mean_coords = np.mean(non_missing_coords, axis=0)
    
    augmented_missing_indices = np.where(coords_augmented == 0)
    coords_augmented[augmented_missing_indices] = mean_coords   
    return coords_augmented

def generate_window_indices(frame_cnt, window_size):
    first_half_window = window_size // 2
    second_half_window = window_size - first_half_window
    window_indices = np.zeros((frame_cnt, window_size), dtype=np.int32)
    for i in range(frame_cnt):
        window_indices[i] = np.arange(i - first_half_window, i + second_half_window)
    window_indices = np.clip(window_indices, 0, frame_cnt - 1)
    return window_indices

def get_missing_mask(coords):
    return (coords == 0).any(axis=(1, -1)).astype(int)

def reorder_by_polygon(coords_window, missing_mask=None):
    # coords_window: (frame_cnt, window_size, pt_cnt, 2)
    missing_cnt = 0
    frame_cnt, window_size = coords_window.shape[:2]
    window_indices = generate_window_indices(frame_cnt, window_size)
    all_radii = get_all_radii(coords_window)
    avg_radii = np.mean(all_radii, axis=1)
    if missing_mask is not None:
        avg_radii[missing_mask == 1] = -1 # don't consider missing frames
        
    prev_order = poly_3(coords_window[0, 0])
    for frame_idx in range(frame_cnt):
        window_radii = avg_radii[window_indices[frame_idx]]
        max_radius_window_idx = np.argmax(window_radii)
        max_radius_frame_idx = window_indices[frame_idx][max_radius_window_idx]
        if avg_radii[max_radius_frame_idx] == -1:
            missing_cnt += 1
            # print(f'window around frame{frame_idx} has no valid points')
            polygon_order = prev_order # use previous polygonorder
        else:
            polygon_order = poly_3(coords_window[max_radius_frame_idx, max_radius_window_idx])
            prev_order = polygon_order
        
        for window_idx in range(window_size):
            coords_window[frame_idx, window_idx] = coords_window[frame_idx, window_idx, polygon_order]
    
    print(f'{missing_cnt} windows have no valid points')
    
    return coords_window