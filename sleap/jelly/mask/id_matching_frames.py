import sys 
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm

def video_loader(video_path, target_size=None, load_as_tensor=False):
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

full_video_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_15min_reencoded_v2.mp4'
clipped_video_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_high_res_5min_track_reencoded_0.mp4'

full_video = video_loader(full_video_path)
clipped_video = video_loader(clipped_video_path)

pad_inx = 150 * 50 * 60
start_search_idx = 0
print(f'pad_inx: {pad_inx}')
print(f'full_video.shape: {full_video.shape}')
print(f'clipped_video.shape: {clipped_video.shape}')

clipped_first_frame = clipped_video[0]
match_idx = None
for i in tqdm(range(start_search_idx, len(full_video))):
    if np.allclose(full_video[i], clipped_first_frame, atol=0.01):
        print(i)
        match_idx = i
        break

print(f'match_idx: {match_idx + pad_inx}')

