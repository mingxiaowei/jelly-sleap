# import sleap
import numpy as np
import cv2
import tensorflow as tf

def load_video(video_path, target_size=None):
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
    return video_array

def create_noisy_points(points_tensor, noise_std=0.05):
    noise = tf.random.normal(tf.shape(points_tensor), stddev=noise_std)
    noisy_points = points_tensor + noise
    
    # Shuffle point identities (axis=1)
    shuffled_points = tf.random.shuffle(tf.transpose(noisy_points, perm=[0, 2, 1]))
    return tf.transpose(shuffled_points, perm=[0, 2, 1])


def load_data():
    video_path = "/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_high_res_10min_track_reencoded.mp4"
    video_data = load_video(video_path)
    
    all_tracked_points = np.load('/home/mingxiao/Desktop/jellyfish/video/video_1_clips/all_points_corrected_10min.npy')
    if video_data.shape[0] > all_tracked_points.shape[0]:
        video_data = video_data[video_data.shape[0] - all_tracked_points.shape[0]:, :, :]
    print(f'video_data.shape: {video_data.shape}')
    print(f'all_tracked_points.shape: {all_tracked_points.shape}')

    video_tensor = tf.constant(video_data, dtype=tf.float16) / 255.0
    video_tensor = tf.expand_dims(video_tensor, axis=-1)  # Add channel dim
    
    frame_cnt, x, y = video_tensor.shape[:3]
    points_tensor = tf.constant(all_tracked_points, dtype=tf.float32)
    canvas_size = tf.constant([y, x], dtype=tf.float32)  # (width, height)
    points_tensor = points_tensor / canvas_size    
    
    noisy_points_tensor = create_noisy_points(points_tensor)
    
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