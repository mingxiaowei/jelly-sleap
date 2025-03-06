import numpy as np
import sleap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# from .animation import get_each_edge_length
import sys
sys.path.append('..')
from eval.eval import calculate_polygon_angles
from animation import get_all_untracked_points_from_lbfs, get_each_edge_length

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

def get_classification_dataset(pred_pts, gt_pts, dist_thres=3):
    """
    Args:
        pred_pts (np.array): (frame_cnt, pt_cnt, 3); predicted points in contiguous frames, 
                             where the last column is prediction score 
        gt_pts (np.array): (frame_cnt, pt_cnt, 2); labeled points in contiguous frames

    Returns:
        features: ((frame_cnt - 1) * pt_cnt, 4); 
                  4 features: prediction score, angle, 2 nn distance (current frame); nn distance (prev frame)
        labels: ((frame_cnt - 1) * pt_cnt, ); 1 = correct, 0 = incorrect 
    """
    pred_pts_score = pred_pts[1:, :, 2]
    pred_pts_angle = calculate_polygon_angles(pred_pts[1:, :, :2])
    pred_pts_2nn_dist = get_2nn_dist_multi_frame(pred_pts[1:, :, :2])
    pred_pts_prev_nn_dist = get_nn_dist_prev_frame(pred_pts[:, :, :2])
    features = np.stack([pred_pts_score, pred_pts_angle, pred_pts_2nn_dist, pred_pts_prev_nn_dist], axis=2)
    labels = get_labels(pred_pts[1:, :, :2], gt_pts[1:], dist_thres)
    nan_mask = np.isnan(features).any(axis=2)
    features = features[~nan_mask].reshape(-1, 4)
    labels = labels[~nan_mask]
    return features, labels

def load_dataset(dist_thres=3):
    cont_start_idx = 479850
    cont_end_idx = cont_start_idx + 9000
    new_4k_dataset_path = '/home/mingxiao/Desktop/jellyfish/label/multifish/multifish_animal_1_v11.slp'
    new_4k_dataset = sleap.load_file(new_4k_dataset_path)
    gt_pts = get_all_untracked_points_from_lbfs(new_4k_dataset.labeled_frames, interpolate=False, reorder=False, use_labeled_only=True)
    new_lbf_indices = np.array([lbf.frame_idx for lbf in new_4k_dataset.labeled_frames])

    predicted_pts_path = '/home/mingxiao/Desktop/jellyfish/label/multifish/predictions/multifish_animal_1_v10_predicted_points.npy'
    predicted_pts = np.load(predicted_pts_path)[new_lbf_indices]
    
    pred_pts_score_path = '/home/mingxiao/Desktop/jellyfish/label/multifish/predictions/multifish_animal_1_v10_predicted_points_with_score.npy'
    pred_pts_score = np.load(pred_pts_score_path)[new_lbf_indices]
    
    cont_indices = (new_lbf_indices >= cont_start_idx) & (new_lbf_indices < cont_end_idx)
    cont_gt_pts = gt_pts[cont_indices]
    cont_pred_pts = predicted_pts[cont_indices]
    cont_pred_pts_score = pred_pts_score[cont_indices]
    
    labels = get_labels(cont_pred_pts, cont_gt_pts)
    print(labels.shape)
    print(f'{int(labels.sum())} out of {2515 * 17} ({labels.sum() / (2515 * 17):.2%}) points are correctly predicted')

    cont_features, cont_labels = get_classification_dataset(cont_pred_pts_score, cont_gt_pts, dist_thres)
    
    feature_cnt = 4
    X = cont_features.reshape(-1, feature_cnt)
    y = cont_labels.flatten()
    print(X.shape, y.shape)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1)
    
    return X_train, X_test, y_train, y_test, X_scaled, y
