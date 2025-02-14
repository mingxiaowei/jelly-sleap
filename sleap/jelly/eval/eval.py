import sleap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('..')
from python.animation import *
from python.postprocess import *

def get_2_nn(point_idx, frame_points):
    distances = np.linalg.norm(frame_points - frame_points[point_idx], axis=1)
    sorted_indices = np.argsort(distances)
    return sorted_indices[1:3]

def check_2_nn(point_idx, n1, n2, total_pt_cnt):
    return ((n1 + 1) % total_pt_cnt != point_idx) + ((n2 - 1) % total_pt_cnt != point_idx)

def count_misconnected_nn(point_idx, frame_points):
    n1, n2 = get_2_nn(point_idx, frame_points)
    total_pt_cnt = frame_points.shape[0]
    return min(check_2_nn(point_idx, n1, n2, total_pt_cnt), check_2_nn(point_idx, n2, n1, total_pt_cnt))

def get_swap_count(tracked_points, count_missing=True) -> int:
    non_missing_pts = tracked_points[tracked_points.sum(axis=1) > 0]
    if count_missing:
        swap_count = len(tracked_points) - len(non_missing_pts)
    else:
        swap_count = 0
    for i in range(len(non_missing_pts)):
        swap_count += count_misconnected_nn(i, non_missing_pts)
    return swap_count

def mask_missing_points(filtered_ranges, swap_cnt_lst, mask_value=np.nan):
    mask = np.zeros(len(swap_cnt_lst), dtype=bool)
    for start, end in filtered_ranges:
        mask[start:end] = True
    swap_cnt_lst_masked = np.where(mask, swap_cnt_lst, mask_value)
    return swap_cnt_lst_masked

def eval_dataset(dataset_path, min_range_length=1, mean_scale=0.7, derivative_thres=5):
    dataset = sleap.load_file(dataset_path)
    print(dataset)
    tracked_points = get_all_tracked_points(dataset, reorder=True, interpolate=False, min_score=0, start_idx=0)
    radii = get_all_radii(tracked_points)
    filtered_ranges = get_expanded_periods(radii, 
                                           min_range_length=min_range_length, 
                                           mean_scale=mean_scale, 
                                           derivative_thres=derivative_thres, 
                                           plot=True, verbose=False)
    
    swap_cnt_lst = []
    for i in range(len(tracked_points)):
        swap_cnt_lst.append(get_swap_count(tracked_points[i], count_missing=False))
    swap_cnt_lst_masked = mask_missing_points(filtered_ranges, swap_cnt_lst)

    plt.figure(figsize=(12, 6))
    plt.plot(swap_cnt_lst_masked)
    plt.xlabel('Frame Number')
    plt.ylabel('Swap Count')
    plt.show()
