import sys
import json
import os
import pandas as pd
import joblib
from matplotlib import animation

sys.path.append('..')
sys.path.append('/Users/mingxiaowei/Desktop/kennedylab/turbulence/jelly-sleap/sleap/jelly/python')
from python.polygon_based_correction import *
# from python.animation import *
from eval.eval import *
from python.dataset_conversion import *
from autoencoder.src.data_loader import *
from correction.interpolate import *


def preprocess(pred_pts, thres=12):
    pred_pts = pred_pts.copy()
    non_missing_mask = get_missing_count(pred_pts) == 0
    frame_cnt, pt_cnt = pred_pts.shape[:2]
    for frame_idx in tqdm(range(1, frame_cnt)):
        non_missing_mask_copy = non_missing_mask.copy()
        
        # 1. remove outliers 
        for pt_idx in range(pt_cnt):
            if 0 in pred_pts[frame_idx, pt_idx]:
                continue
            dist = np.linalg.norm(pred_pts[frame_idx, pt_idx] - pred_pts[frame_idx - 1, pt_idx])
            if dist > thres:
                pred_pts[frame_idx, pt_idx] = 0 # mark as missing
                non_missing_mask_copy[frame_idx, pt_idx] = 0
                
        # 2. interpolate missing points 
        interpolated_pts = avg_flow_interpolate(pred_pts, frame_idx=frame_idx, window_size=4, verbose=False, non_missing_mask=non_missing_mask_copy)
        pred_pts[frame_idx] = interpolated_pts
        
    return pred_pts


def assign_ids_single_frame(curr_pts, ref_pts, thres=15, return_id_mapping=False, verbose=False):
    aligned_pts = np.zeros_like(ref_pts)
    curr_pts_cnt = curr_pts.shape[0]
    ref_pts_cnt = ref_pts.shape[0]
    unused_ref = np.ones(ref_pts_cnt, dtype=bool)
    unused_curr = np.ones(curr_pts_cnt, dtype=bool)
    id_mapping = {}
    
    for curr_idx in range(curr_pts_cnt):
        if 0 in curr_pts[curr_idx]:
            unused_curr[curr_idx] = 0 # remove missing points
    
    while unused_ref.any() and unused_curr.any():
        
        dist_queue = []
        unused_ref_indices = np.where(unused_ref)[0]
        unused_curr_indices = np.where(unused_curr)[0]
        
        for ref_idx in unused_ref_indices:
            curr_pt = ref_pts[ref_idx]
            dists = np.linalg.norm(curr_pts - curr_pt, axis=1)
            match_queue = np.column_stack((dists, np.arange(curr_pts_cnt)))[unused_curr_indices] # (dist, curr_idx)
            min_dist_idx = np.argmin(match_queue[:, 0])
            dist_queue.append((match_queue[min_dist_idx, 0], ref_idx, match_queue[min_dist_idx, 1]))

        dist_queue.sort(key=lambda x: x[0])
        if len(dist_queue) == 0 or dist_queue[0][0] > thres:
            break
        
        for dist, ref_idx, curr_idx in dist_queue:
            ref_idx = int(ref_idx)
            curr_idx = int(curr_idx)
            if dist > thres:
                if verbose:
                    print(f'curr dist = {dist:.2f}, thres = {thres}')
                break
            if unused_ref[ref_idx] and unused_curr[curr_idx]:
                aligned_pts[ref_idx] = curr_pts[curr_idx]
                unused_ref[ref_idx] = 0
                unused_curr[curr_idx] = 0
                id_mapping[ref_idx] = curr_idx
                
    if return_id_mapping:
        return aligned_pts, id_mapping
    else:
        return aligned_pts

def poly_align(curr_frame_pts, prev_frame_pts, poly_constructor=poly_4, align_prev=False):
    curr_frame_pts = curr_frame_pts.copy()
    prev_frame_pts = prev_frame_pts.copy()
    if align_prev:
        prev_frame_poly_order = poly_constructor(prev_frame_pts)
        prev_frame_pts = prev_frame_pts[prev_frame_poly_order]
    curr_frame_poly_order = poly_constructor(curr_frame_pts)
    curr_frame_pts = curr_frame_pts[curr_frame_poly_order]
    best_roll = find_best_roll(curr_frame_pts, prev_frame_pts)
    curr_frame_pts = np.roll(curr_frame_pts, best_roll, axis=0)
    return curr_frame_pts, prev_frame_pts

def preprocess2(pred_pts, thres=18, window_size=4, align_first_frame=True, poly_align_each_step=False):
    pred_pts = pred_pts.copy()
    frame_cnt, pt_cnt = pred_pts.shape[:2]
    if align_first_frame:
        first_frame_poly_order = poly_4(pred_pts[0])
        pred_pts[0] = pred_pts[0, first_frame_poly_order]
    for frame_idx in tqdm(range(1, frame_cnt)):
        
        # 1. align the current frame to the previous frame
        curr_pts, prev_pts = pred_pts[frame_idx], pred_pts[frame_idx - 1]
        curr_pts_aligned = assign_ids_single_frame(curr_pts, prev_pts, thres=thres, return_id_mapping=False)
        pred_pts[frame_idx] = curr_pts_aligned
                
        # 2. interpolate missing points 
        interpolated_pts = avg_flow_interpolate(pred_pts, frame_idx=frame_idx, window_size=window_size, verbose=False)

        # 3. poly_align with the previous frame (optional)
        if poly_align_each_step:
            curr_pts_aligned, _ = poly_align(interpolated_pts, prev_pts, align_prev=False)
            pred_pts[frame_idx] = curr_pts_aligned
        else:
            pred_pts[frame_idx] = interpolated_pts
        
    return pred_pts

def preprocess3(pred_pts, thres=18, align_first_frame=True, poly_align_each_step=False):
    pred_pts = pred_pts.copy()
    frame_cnt, pt_cnt = pred_pts.shape[:2]
    if align_first_frame:
        first_frame_poly_order = poly_4(pred_pts[0])
        pred_pts[0] = pred_pts[0, first_frame_poly_order]
    for frame_idx in tqdm(range(1, frame_cnt)):
        
        # 1. align the current frame to the previous frame
        curr_pts, prev_pts = pred_pts[frame_idx], pred_pts[frame_idx - 1]
        curr_pts_aligned = assign_ids_single_frame(curr_pts, prev_pts, thres=thres, return_id_mapping=False)
        pred_pts[frame_idx] = curr_pts_aligned
                
        # 2. interpolate missing points 
        interpolated_pts = polar_interpolate(pred_pts[frame_idx], verbose=False)
        
        # 3. poly_align with the previous frame
        if poly_align_each_step:
            curr_pts_aligned, _ = poly_align(interpolated_pts, prev_pts, align_prev=False)
            pred_pts[frame_idx] = curr_pts_aligned
        else:
            pred_pts[frame_idx] = interpolated_pts
        
    return pred_pts