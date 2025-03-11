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
from tensorflow.keras import backend as K
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
    
    print(f'swap count: mean = {np.mean(swap_cnt_lst):.2f}, std = {np.std(swap_cnt_lst):.2f}')
    print(f'missing count: mean = {np.mean(missing_cnt_lst):.2f}, std = {np.std(missing_cnt_lst):.2f}')

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

def compare_with_gt(predicted_points, labeled_points, frame_indices, dist_thresh=10):
    assert predicted_points.shape == labeled_points.shape, f'predicted_points.shape: {predicted_points.shape}, labeled_points.shape: {labeled_points.shape}'
    swap_cnt_lst = []
    missing_cnt_lst = []
    frame_cnt, tb_cnt = predicted_points.shape[:2]
    for i in range(frame_cnt):
        swap_cnt = 0
        missing_cnt = 0
        used_nn = []
        for j in range(tb_cnt):
            x, y = predicted_points[i, j]
            # x_gt, y_gt = labeled_points[i, j]
            if np.isnan(x) or np.isnan(y) or x + y == 0:
                missing_cnt += 1
                continue
            diff = labeled_points[i] - predicted_points[i, j]
            dists = np.hypot(diff[:, 0], diff[:, 1])
            min_dist_idx = np.argmin(dists)
            min_dist = dists[min_dist_idx]
            if min_dist > dist_thresh or min_dist_idx in used_nn:
                swap_cnt += 1
            else:
                used_nn.append(min_dist_idx)
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

def calculate_polygon_angles(points):
    if points.ndim == 2:
        return calculate_polygon_angles_single_frame(points)
    else:
        return np.array([calculate_polygon_angles_single_frame(points[i]) for i in range(points.shape[0])])

def calculate_polygon_angles_single_frame(points):
    non_missing_mask = (points.sum(axis=1) > 0) | ~(np.isnan(points).any(axis=1))
    non_missing_indices = np.where(non_missing_mask)[0]
    angles = np.zeros(len(points))
    
    non_missing_cnt = len(non_missing_indices)
    for i in range(non_missing_cnt):
        p1 = points[non_missing_indices[i - 1]]  
        p2 = points[non_missing_indices[i]]      
        p3 = points[non_missing_indices[(i + 1) % non_missing_cnt]]  
        
        v1 = p1 - p2
        v2 = p3 - p2
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        dot_product = np.dot(v1, v2)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))  # Clip for numerical stability
        angle_deg = np.degrees(angle_rad)
        if np.cross(v1, v2) > 0:
            angle_deg = 360 - angle_deg

        angles[non_missing_indices[i]] = angle_deg

    return angles

def get_missing_count(pts):
    frame_cnt, pt_cnt = pts.shape[:2]
    pts = pts[:, :, :2]
    missing_cnt = np.zeros((frame_cnt, pt_cnt))
    for frame_idx in range(frame_cnt):
        for pt_idx in range(pt_cnt):
            if pts[frame_idx, pt_idx, :].sum() == 0 or np.isnan(pts[frame_idx, pt_idx, :]).any():
                missing_cnt[frame_idx, pt_idx] = 1
    return missing_cnt

def animate_classification_results(
        pred_pts: np.ndarray,  
        gt_pts: np.ndarray,  
        checked_labels: np.ndarray,
        x: int = 170,
        y: int = 174,
        fps: int = 50,
        bg_video: Union[sleap.Video, np.array] = None,
        bg_video_start_idx: int = 1,
        output_path: str = None,
        text_dict: dict = None
    ) -> animation.FuncAnimation:
    if text_dict is None:
        text_dict = {0: 'FP', 1: 'TN', 2: 'FN', 3: 'TP', -1: 'nan'}
    if pred_pts.shape[2] == 3:
        pred_pts = pred_pts[:, :, :2]
        
    pred_only = gt_pts is None
    if pred_only:
        gt_pts = pred_pts
        
    assert pred_pts.shape == gt_pts.shape, f'points shape mismatch: {pred_pts.shape} != {gt_pts.shape}'
    
    checked_labels = checked_labels[bg_video_start_idx:]
    pred_pts = pred_pts[bg_video_start_idx:]
    gt_pts = gt_pts[bg_video_start_idx:]
    
    frame_cnt = gt_pts.shape[0]
    n_plots = 1
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_plots, figsize=(8*n_plots, 8))
    if n_plots == 1:
        axes = [axes]  # Make axes iterable when only one subplot
    
    # Initialize plot elements for each subplot
    lines = []
    scats = []
    bg_imgs = []
    texts = []  # Add list to store text annotations
    
    vid_max_val = 255 if isinstance(bg_video, sleap.Video) else 1
    
    for ax in axes:
        ax.set_xlim(0, x)
        ax.set_ylim(0, y)
        ax.invert_yaxis()
        
        # Initialize empty line and scatter objects for each subplot
        line, = ax.plot([], [], 'b-', lw=1)
        pred_scat = ax.scatter([], [], c='red', s=30)
        gt_scat = ax.scatter([], [], c='green', s=30)
        bg_img = ax.imshow(np.zeros((y, x)), cmap='gray', vmin=0, vmax=vid_max_val)
        
        # Initialize empty text annotations
        frame_texts = []
        for _ in range(pred_pts.shape[1]):  # number of points per frame
            txt = ax.text(0, 0, '', fontsize=8, color='blue')
            frame_texts.append(txt)
        
        lines.append(line)
        scats.append([pred_scat, gt_scat])
        bg_imgs.append(bg_img)
        texts.append(frame_texts)

    def init():
        plot_elements = []
        for line, (pred_scat, gt_scat), bg_img, frame_texts in zip(lines, scats, bg_imgs, texts):
            line.set_data([], [])
            pred_scat.set_offsets(np.zeros((0, 2)))
            gt_scat.set_offsets(np.zeros((0, 2)))
            for txt in frame_texts:
                txt.set_position((0, 0))
                txt.set_text('')
            if pred_only:
                plot_elements.extend([pred_scat, bg_img] + frame_texts)
            else:
                plot_elements.extend([pred_scat, gt_scat, bg_img] + frame_texts)
        return plot_elements

    def animate(frame):
        plot_elements = []
        
        # Update background
        if bg_video is not None:
            if isinstance(bg_video, sleap.Video):
                bg_frame = bg_video.get_frame(frame + bg_video_start_idx)[:, :, 0]
            else:
                bg_frame = bg_video[frame + bg_video_start_idx]
            for bg_img in bg_imgs:
                bg_img.set_array(bg_frame)
        
        # Update each subplot
        for i, (pred_pt, gt_pt, checked_label, line, (pred_scat, gt_scat), frame_texts) in enumerate(
            zip([pred_pts], [gt_pts], [checked_labels], lines, scats, texts)):
            
            pred_points = pred_pt[frame]
            gt_points = gt_pt[frame]
            # pred_points_closed = np.vstack([pred_points, pred_points[0]])
            # gt_points_closed = np.vstack([gt_points, gt_points[0]])
            label_check = checked_label[frame]  # Get labels for current frame
            # Update scatter plots
            pred_scat.set_offsets(pred_points)
            gt_scat.set_offsets(gt_points)
            
            # Update text annotations
            for j, (point, is_correct) in enumerate(zip(pred_points, label_check)):
                txt = frame_texts[j]
                txt.set_position((point[0] + 2, point[1] + 2))  # Offset text slightly from point
                txt.set_text(text_dict[is_correct])
            
            plot_elements.extend([pred_scat, gt_scat, bg_imgs[i]] + frame_texts)
            
        return plot_elements

    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                 frames=frame_cnt, interval=50, blit=True)

    # Optional: save animation
    if output_path is not None:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        anim.save(output_path, writer='ffmpeg', fps=fps)
        print(f"Animation saved to {output_path}")
        
    plt.tight_layout()
    return anim