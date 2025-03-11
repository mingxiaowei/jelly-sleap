import numpy as np
import sleap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import make_scorer, recall_score, classification_report
from sklearn.base import clone

# from .animation import get_each_edge_length
import sys
sys.path.append('..')
from eval.eval import calculate_polygon_angles, get_missing_count
from animation import get_all_untracked_points_from_lbfs, get_each_edge_length, get_all_untracked_points

def get_2nn_dist_multi_frame(pts):
    frame_cnt, pt_cnt = pts.shape[:2]
    pts_2nn_dist = np.full((frame_cnt, pt_cnt), -1, dtype=np.float32) # -1 means missing
    for i in range(frame_cnt):
        non_missing_mask = (pts[i].sum(axis=1) > 0) | ~(np.isnan(pts[i]).any(axis=1))
        non_missing_pts = pts[i][non_missing_mask]
        edge_lengths = get_each_edge_length(non_missing_pts) # (pt_cnt,)
        shifted_edge_lengths = np.roll(edge_lengths, 1)
        pts_2nn_dist[i][non_missing_mask] = (edge_lengths + shifted_edge_lengths) / np.mean(edge_lengths) # normalize by avg edge length in the current frame
    return pts_2nn_dist

def get_nn_dist_prev_frame(pts, max_dist=80):
    frame_cnt, pt_cnt = pts.shape[:2]
    pts_prev_nn_dist = np.full((frame_cnt - 1, pt_cnt), max_dist, dtype=np.float32) # if a point does not have a nn in the previous frame, its distance is set to inf
    for frame_idx in range(1, frame_cnt):
        curr_pts = pts[frame_idx]
        prev_pts = pts[frame_idx - 1]
        
        # for each point, find its nearest neighbor in the previous frame
        # a point from the previous frame can have at most one nn in the current frame
        dists_queue = [] # (dist, pt_idx, nn_idx)
        used_nn = []
        for pt_idx in range(pt_cnt):
            all_dists = np.linalg.norm(prev_pts - curr_pts[pt_idx], axis=1)
            min_dist_idx = np.argmin(all_dists)
            dists_queue.append((all_dists[min_dist_idx], pt_idx, min_dist_idx))
        dists_queue.sort(key=lambda x: x[0])
        
        for dist, pt_idx, nn_idx in dists_queue:
            if nn_idx not in used_nn:
                used_nn.append(nn_idx)
                pts_prev_nn_dist[frame_idx - 1, pt_idx] = dist
                
    return pts_prev_nn_dist

def get_labels(pred_pts, gt_pts, dist_thres=3):
    pred_pts = pred_pts[:, :, :2]
    assert pred_pts.shape == gt_pts.shape, f'shape mismatch: {pred_pts.shape} != {gt_pts.shape}'
    frame_cnt, pt_cnt = pred_pts.shape[:2]
    labels = np.zeros((frame_cnt, pt_cnt))
    
    for frame_idx in range(frame_cnt):
        gt_curr_frame_pts = gt_pts[frame_idx]
        dist_queue = [] # (dist, pt_idx, nn_idx)
        used_nn = []
        
        for pt_idx in range(pt_cnt):
            curr_pt = pred_pts[frame_idx, pt_idx]
            dists = np.linalg.norm(gt_curr_frame_pts - curr_pt, axis=1)
            min_dist_idx = np.argmin(dists)
            dist_queue.append((dists[min_dist_idx], pt_idx, min_dist_idx))
            
        dist_queue.sort(key=lambda x: x[0])
        
        for dist, pt_idx, nn_idx in dist_queue:
            if dist > dist_thres:
                break
            if nn_idx not in used_nn:
                used_nn.append(nn_idx)
                labels[frame_idx, pt_idx] = 1
                
    print(f'total point count = {pt_cnt * frame_cnt}, correct = {int(labels.sum())}, incorrect = {(labels == 0).sum()}')
                
    return labels

def get_classification_dataset(pred_pts, gt_pts, dist_thres=3, return_nan_mask=False, load_2nn_dist=True):
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
    features, nan_mask = get_prediction_dataset(pred_pts, return_nan_mask=True, load_2nn_dist=load_2nn_dist)
    labels = get_labels(pred_pts[1:, :, :2], gt_pts[1:], dist_thres)
    labels = labels[~nan_mask]
    if return_nan_mask:
        return features, labels, nan_mask
    else:
        return features, labels

def get_prediction_dataset(pred_pts, return_nan_mask=False, load_2nn_dist=True):
    pred_pts_score = pred_pts[1:, :, 2]
    pred_pts_angle = calculate_polygon_angles(pred_pts[1:, :, :2])
    pred_pts_prev_nn_dist = get_nn_dist_prev_frame(pred_pts[:, :, :2])
    feat_cnt = 3 + load_2nn_dist
    if load_2nn_dist:
        pred_pts_2nn_dist = get_2nn_dist_multi_frame(pred_pts[1:, :, :2])
        features = np.stack([pred_pts_score, pred_pts_angle, pred_pts_2nn_dist, pred_pts_prev_nn_dist], axis=2)
    else:
        features = np.stack([pred_pts_score, pred_pts_angle, pred_pts_prev_nn_dist], axis=2)
    
    nan_mask = np.isnan(features).any(axis=2)
    missing_mask = get_missing_count(pred_pts[1:, :, :2]) > 0 # 1 = missing, 0 = present
    nan_mask = np.logical_or(nan_mask, missing_mask)
    features = features[~nan_mask].reshape(-1, feat_cnt)
    
    if return_nan_mask:
        return features, nan_mask
    else:
        return features

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
    
    return X_train, X_test, y_train, y_test, X_scaled, y, scaler

def augment_pred_pts(pred_pts, augment_rate=0.5):
    pred_pts_augmented = pred_pts[:, :, :2].copy()
    missing_cnt = get_missing_count(pred_pts_augmented)
    selected_frames = np.random.choice(pred_pts_augmented.shape[0], int(pred_pts_augmented.shape[0] * augment_rate), replace=False)

    for frame_idx in selected_frames:
        aug_candidates = np.where(missing_cnt[frame_idx] == 0)[0]
        aug_pt_indices = np.random.choice(aug_candidates, np.random.randint(1, 4), replace=False)
        for pt_idx in aug_pt_indices:
            pred_pts_augmented[frame_idx, pt_idx] += np.random.normal(10, 3, size=2)
    
    return pred_pts_augmented

def load_20s_dataset(dist_thres=3, load_2nn_dist=True, augment_rate=None):
    predicted_dataset_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/correction_test/a1_1h_20s_with_scores.slp'
    predicted_dataset = sleap.load_file(predicted_dataset_path)
    print(predicted_dataset)

    corrected_dataset_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/correction_test/a1_1h_20s_no_scores_corrected.slp'
    corrected_dataset = sleap.load_file(corrected_dataset_path)
    print(corrected_dataset)

    pred_pts_with_scores = get_all_untracked_points(predicted_dataset, 
                                                    interpolate=False, use_labeled_only=False, load_pred_score=True)
    gt_pts = get_all_untracked_points(corrected_dataset, interpolate=False, use_labeled_only=True)
    
    if augment_rate is not None:
        pred_pts_with_scores[2:, :, :2] = augment_pred_pts(pred_pts_with_scores[2:, :, :2], augment_rate)
    
    cont_features, cont_labels = get_classification_dataset(pred_pts_with_scores, gt_pts, dist_thres=dist_thres, load_2nn_dist=load_2nn_dist)
    
    feature_cnt = 3 + load_2nn_dist
    X = cont_features.reshape(-1, feature_cnt)
    y = cont_labels.flatten()
    print(X.shape, y.shape)

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1)
    
    return X_train, X_test, y_train, y_test, X_scaled, y, scaler

def custom_resample(X, y, minority_class=0, minor_to_major_ratio=0.5, dowmsample_majority_ratio=1, verbose=False):
    """
    Resample dataset by undersampling majority class and oversampling minority class
    
    Args:
        X: features array
        y: labels array
        minority_class: the class to oversample (default 0)
        minor_to_major_ratio: desired ratio of minority to majority samples (default 0.5)
        dowmsample_majority_ratio: ratio of majority to minority samples to downsample (default 1; no downsampling)
    """
    # Separate majority and minority classes
    X_majority = X[y != minority_class]
    y_majority = y[y != minority_class]
    X_minority = X[y == minority_class]
    y_minority = y[y == minority_class]
    
    # Calculate desired number of samples
    n_minority = len(X_minority)
    n_majority = len(X_majority)
    desired_minority = int(n_majority * minor_to_major_ratio * dowmsample_majority_ratio)
    
    # Downsample majority class
    if dowmsample_majority_ratio < 1:
        indices = np.random.choice(len(X_majority), int(n_majority * dowmsample_majority_ratio), replace=False)
        X_majority_resampled = X_majority[indices]
        y_majority_resampled = y_majority[indices]
    else:
        X_majority_resampled = X_majority
        y_majority_resampled = y_majority
    
    # Oversample minority class
    if desired_minority > n_minority:
        indices = np.random.choice(len(X_minority), desired_minority, replace=True)
        X_minority_resampled = X_minority[indices]
        y_minority_resampled = y_minority[indices]
    else:
        X_minority_resampled = X_minority
        y_minority_resampled = y_minority
    
    if verbose:
        print(f'majority count: {len(X_majority_resampled)}, minority count: {len(X_minority_resampled)}')
    
    # Combine the datasets
    X_resampled = np.vstack([X_majority_resampled, X_minority_resampled])
    y_resampled = np.hstack([y_majority_resampled, y_minority_resampled])
    
    # Shuffle
    indices = np.arange(len(X_resampled))
    np.random.shuffle(indices)
    
    return X_resampled[indices], y_resampled[indices]

def grid_search_with_resampling_cv(X, y, model, param_grid, 
                                n_splits=5, 
                                minority_class=0, 
                                minor_to_major_ratio=0.2,
                                dowmsample_majority_ratio=0.8):
    """
    Perform grid search with resampling in each fold
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    best_score = 0
    best_params = None
    best_model = None
    
    # Create scorer that focuses on minority class recall
    scorer = make_scorer(recall_score, pos_label=minority_class)
    
    # For each parameter combination
    for params in ParameterGrid(param_grid):
        fold_scores = []
        
        # For each fold
        for train_idx, val_idx in cv.split(X, y):
            # Split data
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            # Resample training data
            X_train_resampled, y_train_resampled = custom_resample(
                X_train_fold, y_train_fold,
                minority_class=minority_class, 
                minor_to_major_ratio=minor_to_major_ratio,
                dowmsample_majority_ratio=dowmsample_majority_ratio
            )
            
            # Train model with current parameters
            model_fold = clone(model)
            model_fold.set_params(**params)
            model_fold.fit(X_train_resampled, y_train_resampled)
            
            # Score on validation set
            score = scorer(model_fold, X_val_fold, y_val_fold)
            fold_scores.append(score)
        
        # Average score across folds
        mean_score = np.mean(fold_scores)
        
        # Update best if improved
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_model = model_fold
            
        print(f"Params: {params}")
        print(f"Mean score: {mean_score:.3f}")
        
        y_pred = model_fold.predict(X_val_fold)
        print(classification_report(y_val_fold, y_pred))
    
    return best_params, best_score, best_model

def grid_search_with_resampling(X_train, y_train, X_test, y_test, model, param_grid, 
                                minority_class=0, 
                                minor_to_major_ratio=0.2,
                                dowmsample_majority_ratio=0.8):
    """
    Perform grid search with resampling in each fold
    """
    best_score = 0
    best_params = None
    best_model = None
    
    X_train_resampled, y_train_resampled = custom_resample(
        X_train, y_train,
        minority_class=minority_class, 
        minor_to_major_ratio=minor_to_major_ratio,
        dowmsample_majority_ratio=dowmsample_majority_ratio
    )
    
    # Create scorer that focuses on minority class recall
    scorer = make_scorer(recall_score, pos_label=minority_class)
    
    # For each parameter combination
    for params in ParameterGrid(param_grid):
        
        # Train model with current parameters
        model_fold = clone(model)
        model_fold.set_params(**params)
        model_fold.fit(X_train_resampled, y_train_resampled)
        
        # Score on validation set
        score = scorer(model_fold, X_test, y_test)
        
        # Update best if improved
        if score > best_score:
            best_score = score
            best_params = params
            best_model = model_fold
            
        print(f"Params: {params}")
        print(f"Score: {score:.3f}")
        
        y_pred = model_fold.predict(X_test)
        print(classification_report(y_test, y_pred))
    
    return best_params, best_score, best_model