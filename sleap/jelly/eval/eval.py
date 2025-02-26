import sleap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
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

def count_misconnected_nn(point_idx, frame_points, binarize=True, method='nn', polygon_constructor=poly_4):
    total_pt_cnt = frame_points.shape[0]
    if method == 'nn':
        n1, n2 = get_2_nn(point_idx, frame_points)
    elif method == 'polygon':
        polygon_order = polygon_constructor(frame_points)
        n1, n2 = polygon_order[point_idx - 1], polygon_order[(point_idx + 1) % total_pt_cnt]
        point_idx = polygon_order[point_idx]
    else:
        raise ValueError(f'Unknown method: {method}')
    
    misconnected_cnt = min(check_2_nn(point_idx, n1, n2, total_pt_cnt), check_2_nn(point_idx, n2, n1, total_pt_cnt))
    if binarize:
        return misconnected_cnt > 0
    else:
        return misconnected_cnt

def get_err_count(tracked_points, count_missing=True, method='nn', binarize=True, polygon_constructor=poly_4) -> int:
    if np.any(np.isnan(tracked_points)):
        non_missing_pts = tracked_points[~np.isnan(tracked_points).any(axis=1)]
    else:
        non_missing_pts = tracked_points[tracked_points.sum(axis=1) > 0]
    swap_count = 0
    for i in range(len(non_missing_pts)):
        swap_count += count_misconnected_nn(i, non_missing_pts, 
                                            method=method, binarize=binarize, polygon_constructor=polygon_constructor)
    if count_missing:
        missing_count = len(tracked_points) - len(non_missing_pts)
        return swap_count, missing_count
    return swap_count

def get_mask_from_ranges(filtered_ranges, frame_cnt):
    mask = np.zeros(frame_cnt, dtype=bool)
    for start, end in filtered_ranges:
        mask[start:end] = True
    return mask

def mask_non_expanded_frames(filtered_ranges, swap_cnt_lst, mask_value=np.nan):
    mask = get_mask_from_ranges(filtered_ranges, len(swap_cnt_lst))
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
                              count_missing=True,
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
                             count_missing=True, 
                             polygon_constructor=poly_4, 
                             save_path=None):
    radii = get_all_radii(tracked_points)
    filtered_ranges = get_expanded_periods(radii, 
                                           min_range_length=min_range_length, 
                                           mean_scale=mean_scale, 
                                           derivative_thres=derivative_thres, 
                                           plot=True, verbose=verbose, use_mean=use_mean)
    
    swap_cnt_lst = []
    missing_cnt_lst = []
    expansion_mask = get_mask_from_ranges(filtered_ranges, len(tracked_points))
    fill_val = np.nan
    for i in tqdm(range(len(tracked_points))):
        if expansion_mask[i]:
            swap_cnt, missing_cnt = get_err_count(tracked_points[i], 
                                                  count_missing=count_missing, method=method, binarize=binarize, polygon_constructor=polygon_constructor)
            swap_cnt_lst.append(swap_cnt)
            missing_cnt_lst.append(missing_cnt)
        else:
            swap_cnt_lst.append(fill_val)
            missing_cnt_lst.append(fill_val)
    
    if save_path is not None:
        parent_dir = os.path.dirname(save_path)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        np.save(save_path, swap_cnt_lst)

    plot_error_count_over_time(swap_cnt_lst, missing_cnt_lst, plot_total=False, tb_cnt=17)
    plot_error_count_distribution(swap_cnt_lst, missing_cnt_lst, tb_cnt=17)
    
    return swap_cnt_lst, missing_cnt_lst,filtered_ranges

def plot_error_count_over_time(swap_cnt_lst, missing_cnt_lst, plot_total=False, tb_cnt=17):
    plt.figure(figsize=(10, 5))
    plt.plot(swap_cnt_lst, label='Swap Count')
    plt.plot(missing_cnt_lst, label='Missing Count')
    if plot_total:
        plt.plot(swap_cnt_lst + missing_cnt_lst, label='Total Error Count')
    plt.legend()
    plt.xlabel('Frame Number')
    plt.ylabel('Error Count')
    plt.yticks(range(tb_cnt + 1))
    plt.show()

def plot_error_count_distribution(swap_cnt_lst, missing_cnt_lst, tb_cnt=17):
    plt.hist(swap_cnt_lst, label='Swap Count', bins=tb_cnt + 1)
    plt.hist(missing_cnt_lst, label='Missing Count', bins=tb_cnt + 1)
    plt.legend()
    plt.xlabel('Error Count')
    plt.ylabel('Frame Count')
    plt.xticks(range(tb_cnt + 1))
    plt.title('Error Count Distribution')
    plt.show()

def eval_discrete_points(tracked_points, frame_indices,
                         binarize=True, 
                         method='nn',
                         count_missing=True, 
                         polygon_constructor=poly_4):
    
    tb_cnt = tracked_points.shape[1]
    swap_cnt_lst = []
    missing_cnt_lst = []
    for i in tqdm(range(len(tracked_points))):
        swap_cnt, missing_cnt = get_err_count(tracked_points[i], 
                                              count_missing=count_missing, 
                                              method=method, 
                                              binarize=binarize, 
                                              polygon_constructor=polygon_constructor)
        swap_cnt_lst.append(swap_cnt)
        missing_cnt_lst.append(missing_cnt)

    plt.figure(figsize=(10, 5))
    plt.scatter(frame_indices, swap_cnt_lst, s=10, label='Swap Count')
    plt.scatter(frame_indices, missing_cnt_lst, s=10, label='Missing Count')
    plt.legend()
    plt.xlabel('Frame Number')
    plt.ylabel('Error Count')
    plt.yticks(range(tb_cnt + 1))
    plt.show()
    
    plot_error_count_distribution(swap_cnt_lst, missing_cnt_lst, tb_cnt=tb_cnt)
    
    return swap_cnt_lst, missing_cnt_lst
