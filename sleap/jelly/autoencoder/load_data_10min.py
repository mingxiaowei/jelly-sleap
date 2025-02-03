# import sleap
import numpy as np
import cv2
import tensorflow as tf

video_path = "/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_high_res_10min_track_reencoded.mp4"
points_path = "/home/mingxiao/Desktop/jellyfish/video/video_1_clips/all_points_corrected_10min.npy"

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

def load_prediction_input(video_path=video_path, points_path=points_path, load_as_tensor=True):
    video_tensor = load_video(video_path, load_as_tensor=load_as_tensor)
    points_tensor = load_points(points_path, load_as_tensor=load_as_tensor)
    
    video_tensor, _, points_tensor = preprocess_data(points_tensor, video_tensor, add_noise=False)
    return video_tensor, points_tensor

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

def load_data_cpu():
    with tf.device('/cpu:0'):
        train_data, val_data = load_data()
    return train_data, val_data

def load_prediction_input_cpu():
    with tf.device('/cpu:0'):
        prediction_input = load_prediction_input()
    return prediction_input