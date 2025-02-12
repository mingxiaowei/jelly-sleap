import numpy as np
import cv2
import tensorflow as tf

video_path = "/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_high_res_10min_track_reencoded.mp4"
corrected_points_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/manual_5min_c0_points.npy'
raw_points_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_raw_points.npy'

def load_points(raw_points_path=raw_points_path, corrected_points_path=corrected_points_path):
    coords_corrected = np.load(corrected_points_path).astype('float32')
    coords_raw = np.load(raw_points_path).astype('float32')
    
    # Replace (0,0) coordinates with (nan, nan)
    coords_corrected[(coords_corrected == 0).all(axis=2)] = np.nan
    coords_raw[(coords_raw == 0).all(axis=2)] = np.nan
    
    return coords_raw, coords_corrected

def load_video(video_path=video_path, target_size=None, load_as_tensor=False):
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

def get_windows_wrapper(coords_list, window_size=5, normalizer=(170, 174)):
    coords_windows = []
    for coords in coords_list:
        coords_normalized = coords / normalizer
        coords_flat = coords_normalized.reshape((len(coords), -1))
        coords_windows.append(get_sliding_windows(coords_flat, window_size=window_size))
    return coords_windows

def get_dropout_mask(frame_cnt, pt_cnt, dropout_rate=0.01):
    mask = np.zeros((frame_cnt, pt_cnt))
    total_elements = frame_cnt * pt_cnt
    num_ones = int(total_elements * dropout_rate)
    indices = np.random.choice(total_elements, num_ones, replace=False)
    rows = indices // pt_cnt
    cols = indices % pt_cnt
    mask[rows, cols] = 1
    return mask

def augment_points(coords, dropout_rate=0.01, swap_rate=0.005):
    frame_cnt, pt_cnt = coords.shape[:2]
    # 1. dropout
    dropout_mask = get_dropout_mask(frame_cnt, pt_cnt, dropout_rate)
    coords[dropout_mask == 1] = np.nan
    # 2. swap
    swap_frames = np.random.choice(frame_cnt, int(frame_cnt * swap_rate), replace=False)
    for frame_idx in swap_frames:
        swap_pts = np.random.choice(pt_cnt, np.random.randint(2, 5), replace=False)
        swap_pts_shuffled = np.random.permutation(swap_pts)
        coords[frame_idx, swap_pts] = coords[frame_idx, swap_pts_shuffled]
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
    window_size=5,
    load_as_tensor=False,
    shuffle=True,
    augment=False, 
    dropout_rate=0.01,
    swap_rate=0.005, 
    train_size=0.9,
    ):
    coords_raw, coords_corrected = load_points(raw_points_path=raw_points_path, 
                                               corrected_points_path=corrected_points_path)
    frame_cnt, pt_cnt = coords_corrected.shape[:2]
    # 1. augment
    if augment:
        coords_augmented= augment_points(coords_corrected.copy(), dropout_rate, swap_rate)
    # 2. get window
    coords_corrected_windows, coords_augmented_windows = get_windows_wrapper([coords_corrected, coords_augmented], window_size)
    # 3. shuffle
    indices = np.arange(len(coords_corrected_windows))
    if shuffle:
        np.random.shuffle(indices)
    coords_corrected_windows = coords_corrected_windows[indices]
    coords_augmented_windows = coords_augmented_windows[indices]
    # 4. train val split
    X_train, X_val, y_train, y_val = split_train_val(coords_augmented_windows, coords_corrected_windows, 
                                                     train_size=train_size)
    
    if load_video:
        video = load_video(video_path=video_path, target_size=None, load_as_tensor=load_as_tensor)
        video = video[indices]
        video_windows = get_sliding_windows(video, window_size=window_size)
        video_windows = video_windows[indices]
        video_train, video_val = split_train_val(video_windows, train_size=train_size)
        X_train = [video_train, X_train]
        X_val = [video_val, X_val]
        y_train = [video_train, y_train]
        y_val = [video_val, y_val]
        
    return X_train, X_val, y_train, y_val
