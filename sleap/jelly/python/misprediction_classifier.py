import numpy as np
from .animation import get_each_edge_length
import sys
sys.path.append('..')
from eval.eval import calculate_polygon_angles

def get_2nn_dist_multi_frame(pts):
    frame_cnt, pt_cnt = pts.shape[:2]
    pts_2nn_dist = np.zeros((frame_cnt, pt_cnt))
    for i in range(frame_cnt):
        edge_lengths = get_each_edge_length(pts[i]) # (pt_cnt,)
        shifted_edge_lengths = np.roll(edge_lengths, 1)
        pts_2nn_dist[i] = (edge_lengths + shifted_edge_lengths) / np.mean(edge_lengths) # normalize by avg edge length in the current frame
    return pts_2nn_dist

def get_nn_dist_prev_frame(pts):
    frame_cnt, pt_cnt = pts.shape[:2]
    pts_prev_nn_dist = np.zeros((frame_cnt - 1, pt_cnt))
    for frame_idx in range(1, frame_cnt):
        curr_pts = pts[frame_idx]
        prev_pts = pts[frame_idx - 1]
        # for each point, find its nearest neighbor in the previous frame
        # does it matter if two points have the same nearest neighbor?
        # uh let's assume it doesn't matter for now
        for pt_idx in range(pt_cnt):
            all_dists = np.linalg.norm(prev_pts - curr_pts[pt_idx], axis=1)
            pts_prev_nn_dist[frame_idx - 1, pt_idx] = np.min(all_dists)
    return pts_prev_nn_dist

def get_labels(pred_pts, gt_pts, dist_thres=3):
    assert pred_pts.shape == gt_pts.shape, f'shape mismatch: {pred_pts.shape} != {gt_pts.shape}'
    frame_cnt, pt_cnt = pred_pts.shape[:2]
    labels = np.zeros((frame_cnt, pt_cnt))
    for frame_idx in range(frame_cnt):
        gt_curr_frame_pts = gt_pts[frame_idx]
        for pt_idx in range(pt_cnt):
            curr_pt = pred_pts[frame_idx, pt_idx]
            min_dist = np.min(np.linalg.norm(gt_curr_frame_pts - curr_pt, axis=1))
            labels[frame_idx, pt_idx] = min_dist < dist_thres
    return labels

def get_classification_dataset(pred_pts, gt_pts):
    """
    Args:
        pred_pts (np.array): (frame_cnt, pt_cnt, 3); predicted points in contiguous frames, 
                             where the last column is prediction score 
        gt_pts (np.array): (frame_cnt, pt_cnt, 2); labeled points in contiguous frames

    Returns:
        features: (frame_cnt - 1, pt_cnt, 4); 
                  4 features: prediction score, angle, 2 nn distance (current frame); nn distance (prev frame)
        labels: (frame_cnt - 1, ); 1 = correct, 0 = incorrect 
    """
    pred_pts_score = pred_pts[1:, :, 2]
    pred_pts_angle = calculate_polygon_angles(pred_pts[1:, :, :2])
    pred_pts_2nn_dist = get_2nn_dist_multi_frame(pred_pts[1:, :, :2])
    pred_pts_prev_nn_dist = get_nn_dist_prev_frame(pred_pts[:, :, :2])
    features = np.stack([pred_pts_score, pred_pts_angle, pred_pts_2nn_dist, pred_pts_prev_nn_dist], axis=2)
    labels = get_labels(pred_pts[1:, :, :2], gt_pts[1:])
    return features, labels