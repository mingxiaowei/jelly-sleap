# import sleap
import numpy as np
import cv2
import tensorflow as tf

video_path = "/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_high_res_10min_track_reencoded.mp4"
# points_path = "/home/mingxiao/Desktop/jellyfish/video/video_1_clips/all_points_corrected_10min.npy"
points_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/all_tracked_points_raw_0.npy'

tf.device('/cpu:0')

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
        return video_array

def create_noisy_points(points_tensor, noise_std=0.05):
    noise = tf.random.normal(tf.shape(points_tensor), stddev=noise_std)
    noisy_points = points_tensor + noise
    
    # Shuffle point identities (axis=1)
    shuffled_points = tf.random.shuffle(tf.transpose(noisy_points, perm=[0, 2, 1]))
    return tf.transpose(shuffled_points, perm=[0, 2, 1])

def load_points(points_path=points_path, load_as_tensor=False):
    points_array = np.load(points_path)
    if load_as_tensor:
        points_tensor = tf.constant(points_array, dtype=tf.float32)
        return points_tensor
    else:
        return points_array

def preprocess_data(points_tensor, video_tensor, add_noise=True):
    if video_tensor.shape[0] > points_tensor.shape[0]:
        video_tensor = video_tensor[video_tensor.shape[0] - points_tensor.shape[0]:, :, :]
    _, x, y = video_tensor.shape[:3]
    canvas_size = tf.constant([y, x], dtype=tf.float32)  # (width, height)
    points_tensor = points_tensor / canvas_size
    
    if add_noise:
        noisy_points_tensor = create_noisy_points(points_tensor)
    else:
        noisy_points_tensor = points_tensor
    
    return video_tensor, noisy_points_tensor, points_tensor

def load_prediction_input(video_path=video_path, points_path=points_path, load_as_tensor=True, use_cpu=True):
    if use_cpu:
        with tf.device('/cpu:0'):
            video_tensor = load_video(video_path, load_as_tensor=load_as_tensor)
            points_tensor = load_points(points_path, load_as_tensor=load_as_tensor)
            
        video_tensor, _, points_tensor = preprocess_data(points_tensor, video_tensor, add_noise=False)
    else:
        video_tensor = load_video(video_path, load_as_tensor=load_as_tensor)
        points_tensor = load_points(points_path, load_as_tensor=load_as_tensor)
        video_tensor, _, points_tensor = preprocess_data(points_tensor, video_tensor, add_noise=False)
    return video_tensor, points_tensor

def load_prediction_input_pts_only(points_path=points_path, load_as_tensor=True):
    points_tensor = load_points(points_path, load_as_tensor=load_as_tensor)
    return points_tensor

def load_data(video_path=video_path, points_path=points_path, load_as_tensor=True):
    
    video_tensor = load_video(video_path, load_as_tensor=load_as_tensor)
    points_tensor = load_points(points_path, load_as_tensor=load_as_tensor)
    
    video_tensor, noisy_points_tensor, points_tensor = preprocess_data(points_tensor, video_tensor, add_noise=True)
    
    # Calculate split index (e.g., 90% train, 10% validation)
    split_idx = int(0.9 * len(video_tensor))  # Assuming video_tensor has 90003 frames

    # Training data
    train_video = video_tensor[:split_idx]
    train_noisy = noisy_points_tensor[:split_idx]
    train_true = points_tensor[:split_idx]

    # Validation data
    val_video = video_tensor[split_idx:]
    val_noisy = noisy_points_tensor[split_idx:]
    val_true = points_tensor[split_idx:]
    
    batch_size = 64

    # Training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        ((train_video, train_noisy), train_true)
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices(
        ((val_video, val_noisy), val_true)
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset

def load_data_cpu(load_video=True):
    with tf.device('/cpu:0'):
        if load_video:
            train_data, val_data = load_data()
        else:
            train_data, val_data = load_data_pts_only()
    return train_data, val_data

def load_prediction_input_cpu(load_video=True):
    with tf.device('/cpu:0'):
        if load_video:
            prediction_input = load_prediction_input()
        else:
            prediction_input = load_prediction_input_pts_only()
    return prediction_input

def load_data_pts_only(points_path=points_path, load_as_tensor=True):
    
    points_tensor = load_points(points_path, load_as_tensor=load_as_tensor)
    
    # Calculate split index (e.g., 90% train, 10% validation)
    split_idx = int(0.9 * len(points_tensor))  # Assuming video_tensor has 90003 frames

    # Training data
    train_points = points_tensor[:split_idx]
    train_true = points_tensor[:split_idx]

    # Validation data
    val_points = points_tensor[split_idx:]
    val_true = points_tensor[split_idx:]
    
    batch_size = 64

    # Training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_points, train_true)
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_points, val_true)
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset