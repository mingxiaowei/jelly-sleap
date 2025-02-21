import sleap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import sys
sys.path.append('..')
from python.animation import *
from python.postprocess import *
from python.polygon_based_correction import *

def get_2_nn(point_idx, frame_points):
    distances = np.linalg.norm(frame_points - frame_points[point_idx], axis=1)
    sorted_indices = np.argsort(distances)
    return sorted_indices[1:3]

def check_2_nn(point_idx, n1, n2, total_pt_cnt): 
    # return the number of misconnected 2-nn
    return ((n1 + 1) % total_pt_cnt != point_idx) + ((n2 - 1) % total_pt_cnt != point_idx)

def count_misconnected_nn(point_idx, frame_points, binarize=True, method='nn'):
    total_pt_cnt = frame_points.shape[0]
    if method == 'nn':
        n1, n2 = get_2_nn(point_idx, frame_points)
    elif method == 'polygon':
        polygon_order = poly_3(frame_points)
        n1, n2 = polygon_order[point_idx - 1], polygon_order[(point_idx + 1) % total_pt_cnt]
        point_idx = polygon_order[point_idx]
    else:
        raise ValueError(f'Invalid method: {method}')
    
    misconnected_cnt = min(check_2_nn(point_idx, n1, n2, total_pt_cnt), check_2_nn(point_idx, n2, n1, total_pt_cnt))
    if binarize:
        return misconnected_cnt > 0
    else:
        return misconnected_cnt

def get_swap_count(tracked_points, count_missing=True, method='nn', binarize=True) -> int:
    if np.any(np.isnan(tracked_points)):
        non_missing_pts = tracked_points[~np.isnan(tracked_points).any(axis=1)]
    else:
        non_missing_pts = tracked_points[tracked_points.sum(axis=1) > 0]
    if count_missing:
        swap_count = len(tracked_points) - len(non_missing_pts)
    else:
        swap_count = 0
    for i in range(len(non_missing_pts)):
        swap_count += count_misconnected_nn(i, non_missing_pts, method=method, binarize=binarize)
    return swap_count

def mask_missing_points(filtered_ranges, swap_cnt_lst, mask_value=np.nan):
    mask = np.zeros(len(swap_cnt_lst), dtype=bool)
    for start, end in filtered_ranges:
        mask[start:end] = True
    swap_cnt_lst_masked = np.where(mask, swap_cnt_lst, mask_value)
    return swap_cnt_lst_masked

def get_all_radii(tracked_points):
    all_radii = np.zeros(tracked_points.shape[:2])
    for i in range(tracked_points.shape[0]):
        center_pos = np.mean(tracked_points[i], axis=0)
        for j in range(tracked_points.shape[1]):
            all_radii[i, j] = np.linalg.norm(tracked_points[i, j] - center_pos)
    return all_radii

def eval_dataset(dataset_path, 
                 min_range_length=1, 
                 mean_scale=0.7, 
                 derivative_thres=5, 
                 tracked_points_path=None, 
                 use_mean=True, 
                 verbose=False,
                 binarize=True,
                 method='nn', 
                 use_cached_pts=True,
                 count_missing=True):
    if tracked_points_path is None:
        tracked_points_path = dataset_path.replace('.slp', '_tracked_points.npy')
    if os.path.exists(tracked_points_path) and use_cached_pts:
        tracked_points = np.load(tracked_points_path)
    else:
        dataset = sleap.load_file(dataset_path)
        print(f'Getting all tracked points from {dataset}')
        tracked_points = get_all_tracked_points(dataset, reorder=True, interpolate=False, min_score=0, start_idx=0)
        np.save(tracked_points_path, tracked_points)
    return eval_dataset_from_points(tracked_points, 
                                    min_range_length, mean_scale, 
                                    derivative_thres, use_mean, 
                                    verbose, binarize, 
                                    method, count_missing)
    
def eval_dataset_single_model(dataset_path, 
                              min_range_length=1, 
                              mean_scale=0.7, 
                              derivative_thres=5, 
                              tracked_points_path=None, 
                              use_mean=True, 
                              verbose=False,
                              binarize=True,
                              method='nn'):
    if tracked_points_path is None:
        tracked_points_path = dataset_path.replace('.slp', '_tracked_points.npy')
    if os.path.exists(tracked_points_path):
        tracked_points = np.load(tracked_points_path)
    else:
        dataset = sleap.load_file(dataset_path)
        print(f'Getting all tracked points from \n{dataset}')
        tracked_points = get_all_tracked_points_single_model(dataset)
        np.save(tracked_points_path, tracked_points)
    return eval_dataset_from_points(tracked_points, 
                                    min_range_length, mean_scale, 
                                    derivative_thres, use_mean, 
                                    verbose, binarize, 
                                    method, count_missing)
    
def eval_dataset_from_points(tracked_points, 
                             min_range_length=1, 
                             mean_scale=0.7, 
                             derivative_thres=5, 
                             use_mean=True, 
                             verbose=False, 
                             binarize=True, 
                             method='nn',
                             count_missing=True):
    radii = get_all_radii(tracked_points)
    filtered_ranges = get_expanded_periods(radii, 
                                           min_range_length=min_range_length, 
                                           mean_scale=mean_scale, 
                                           derivative_thres=derivative_thres, 
                                           plot=True, verbose=verbose, use_mean=use_mean)
    
    swap_cnt_lst = []
    for i in range(len(tracked_points)):
        swap_cnt_lst.append(get_swap_count(tracked_points[i], count_missing=count_missing, method=method, binarize=binarize))
    swap_cnt_lst_masked = mask_missing_points(filtered_ranges, swap_cnt_lst)

    plt.figure(figsize=(12, 6))
    plt.plot(swap_cnt_lst_masked)
    plt.xlabel('Frame Number')
    plt.ylabel('Swap Count')
    plt.show()
    
    return swap_cnt_lst_masked, filtered_ranges
