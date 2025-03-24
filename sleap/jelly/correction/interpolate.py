import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('/home/mingxiao/Desktop/jelly-sleap/sleap/jelly/python')
from polygon_based_correction import poly_4


def get_missing_count(pts):
    is_2d = pts.ndim == 2
    if is_2d:
        pts = pts[np.newaxis, :, :]
    frame_cnt, pt_cnt = pts.shape[:2]
    pts = pts[:, :, :2]
    missing_cnt = np.zeros((frame_cnt, pt_cnt), dtype=int)
    for frame_idx in range(frame_cnt):
        for pt_idx in range(pt_cnt):
            # if pts[frame_idx, pt_idx, :].sum() == 0 or np.isnan(pts[frame_idx, pt_idx, :]).any():
            if np.isnan(pts[frame_idx, pt_idx, :]).any() or np.isclose(pts[frame_idx, pt_idx, :], 0, atol=1e-2).any():
                missing_cnt[frame_idx, pt_idx] = 1
    if is_2d:
        return missing_cnt[0]
    else:
        return missing_cnt

def cart2pol(xy_arr, center_pos=(0, 0)):
    xy_arr = xy_arr - center_pos
    x = xy_arr[:, 0]
    y = xy_arr[:, 1]
    rho = np.hypot(x, y)  
    phi = np.arctan2(y, x)
    return np.array([rho, phi]).T

def pol2cart(rho_phi_arr, center_pos=(0, 0)):
    rho = rho_phi_arr[:, 0]
    phi = rho_phi_arr[:, 1]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y]).T + center_pos

def polar_interpolate(pts, frame_idx=None, non_missing_mask=None, center_pos=None, verbose=True):
    if pts.ndim == 2:
        return polar_interpolate_single_frame(pts, frame_idx, non_missing_mask, center_pos, verbose)
    elif pts.ndim == 3:
        frame_cnt = pts.shape[0]
        interpolated_pts = np.zeros_like(pts)
        for frame_idx in range(frame_cnt):
            interpolated_pts[frame_idx] = polar_interpolate_single_frame(pts[frame_idx], frame_idx, 
                                                                         non_missing_mask[frame_idx] if non_missing_mask is not None else None, 
                                                                         center_pos, verbose)
        return interpolated_pts
    else:
        raise ValueError(f'pts.ndim must be 2 or 3, but got {pts.ndim}')
    
def polar_interpolate_single_frame(curr_frame_pts, frame_idx=None, non_missing_mask=None, center_pos=None, verbose=True):
    
    if non_missing_mask is None:
        non_missing_mask = get_missing_count(curr_frame_pts) == 0
    
    missing_count = (~non_missing_mask).sum()
    if missing_count > 0:
        print(f'missing count = {missing_count} at frame {frame_idx}')
    
    first_non_missing_idx = np.where(non_missing_mask)[0][0]
    curr_frame_pts = np.roll(curr_frame_pts, -first_non_missing_idx, axis=0)
    non_missing_mask = np.roll(non_missing_mask, -first_non_missing_idx, axis=0)
    non_missing_indices = np.where(non_missing_mask)[0]
    
    if center_pos is None:
        center_pos = curr_frame_pts[non_missing_mask].mean(axis=0)
    
    polar_coords = cart2pol(curr_frame_pts, center_pos)
    polar_coords_deg = polar_coords.copy()
    polar_coords_deg[:, 1] = np.rad2deg(polar_coords_deg[:, 1])
    cart_interpolated_pts = np.zeros_like(curr_frame_pts)
    pt_cnt = len(curr_frame_pts)
    
    for pt_idx in range(pt_cnt):
        if non_missing_mask[pt_idx]:
            cart_interpolated_pts[pt_idx] = curr_frame_pts[pt_idx]
            continue
            
        prev_idx = non_missing_indices[non_missing_indices < pt_idx][-1] if any(non_missing_indices < pt_idx) else non_missing_indices[-1]
        next_idx = non_missing_indices[non_missing_indices > pt_idx][0] if any(non_missing_indices > pt_idx) else non_missing_indices[0]
        
        if next_idx > prev_idx:
            denom = next_idx - prev_idx
        else:
            denom = pt_cnt - (prev_idx - next_idx)
        weight = ((pt_idx - prev_idx) % pt_cnt) / denom
        r1, theta1 = polar_coords[prev_idx]
        r2, theta2 = polar_coords[next_idx]
        if verbose:
            print(f'prev_idx: {prev_idx}, next_idx: {next_idx}')
            print(f'r1: {r1}, theta1: {np.rad2deg(theta1)}')
            print(f'r2: {r2}, theta2: {np.rad2deg(theta2)}')
            print(f'weight: {weight}')

        r_interp = r1 + weight * (r2 - r1)
        theta_diff = (theta2 - theta1)
        if theta_diff < 0:
            theta_diff += 2*np.pi
        theta_interp = theta1 + weight * theta_diff
        if verbose:
            print(f'theta1: {np.rad2deg(theta1)}, theta2: {np.rad2deg(theta2)}')

        cart_interpolated_pts[pt_idx] = pol2cart(np.array([[r_interp, theta_interp]]), center_pos)[0]
        
        if verbose:
            n1, n2 = pol2cart(np.array([[r1, theta1], [r2, theta2]]), center_pos)
            print(f'r_interp: {r_interp}, theta_interp: {np.rad2deg(theta_interp)}')
            print(f'n1: {n1}, n2: {n2}')
            print(f'cart_interpolated_pts: {cart_interpolated_pts[pt_idx]}')

    cart_interpolated_pts = np.roll(cart_interpolated_pts, first_non_missing_idx, axis=0)
    return cart_interpolated_pts

def plot_pred_vs_gt_pts(pred_pts, gt_pts, center_pos=None, canvas_size=(170, 170)):
    assert pred_pts.shape == gt_pts.shape, f'shape mismatch: {pred_pts.shape} != {gt_pts.shape}'
    if center_pos is None:
        center_pos = gt_pts.mean(axis=0)
        
    plt.axis('equal')
    plt.scatter(gt_pts[:, 0], gt_pts[:, 1], color='green', label='true')
    plt.scatter(pred_pts[:, 0], pred_pts[:, 1], color='blue', label='interpolated')
    plt.scatter(center_pos[0], center_pos[1], color='red')
    
    for i, (x, y) in enumerate(gt_pts):
        plt.annotate(str(i), (x, y), 
                    xytext=(5, 5),  # 5 points offset
                    textcoords='offset points',
                    fontsize=8)
     
    plt.legend(fontsize=8, bbox_to_anchor=(1, 1))
    plt.axis('equal')
    plt.show()
    
def test_single_frame_polar_interpolation(gt_pts, frame_idx):
    curr_frame_pts = gt_pts[frame_idx]
    poly_order = poly_4(curr_frame_pts)
    curr_frame_pts = curr_frame_pts[poly_order]
    interp_pts = np.zeros_like(curr_frame_pts)
    for pt_idx in range(curr_frame_pts.shape[0]):
        curr_frame_pts_copy = curr_frame_pts.copy()
        curr_frame_pts_copy[pt_idx] = (0, 0)
        interp_pts[pt_idx] = polar_interpolate(curr_frame_pts_copy, verbose=False)[pt_idx]
    plot_pred_vs_gt_pts(interp_pts, curr_frame_pts)
    mean_err = np.linalg.norm(interp_pts - curr_frame_pts, axis=1).mean()
    print(f'mean error (pixel): {mean_err:.2f}')

def eval_avg_flow_interpolation(pts, window_size=2, reorder=True, verbose=False):
    """
    Evaluate the interpolation performance of a given interpolator.
    
    Args:
        pts: The points to interpolate.
        gt_pts: The ground truth points.
        
    Returns:
        The interpolation error. (frame_cnt, pt_cnt)
    """
    pts = pts[:, :, :2]
    frame_cnt, pt_cnt = pts.shape[:2]
    interpolation_err = np.zeros((frame_cnt, pt_cnt))
    
    for frame_idx in range(frame_cnt):
        if verbose: 
            print(f'frame_idx: {frame_idx}')
        curr_frame_pts = pts[frame_idx]
        gt_pts_copy = pts.copy()
        if reorder:
            polygon_order = poly_4(curr_frame_pts)
            curr_frame_pts = curr_frame_pts[polygon_order]
            gt_pts_copy = gt_pts_copy[:, polygon_order]
        center_pos = curr_frame_pts.mean(axis=0)
        
        for pt_idx in range(pt_cnt):
            
            curr_pt = gt_pts_copy[frame_idx, pt_idx].copy()
            gt_pts_copy[frame_idx, pt_idx] = (0, 0)
            curr_pt_interpolated = avg_flow_interpolate(gt_pts_copy, frame_idx, window_size=window_size, center_pos=center_pos, verbose=False)[pt_idx]
            assert curr_pt.shape == curr_pt_interpolated.shape, f'shape mismatch: {curr_pt.shape} != {curr_pt_interpolated.shape}'
            interpolation_err[frame_idx, pt_idx] = np.linalg.norm(curr_pt - curr_pt_interpolated)
            gt_pts_copy[frame_idx, pt_idx] = curr_pt
            if verbose:
                print(f'curr_pt: {np.round(curr_pt, 2)}, curr_pt_interpolated: {np.round(curr_pt_interpolated, 2)}, err: {np.round(interpolation_err[frame_idx, pt_idx], 2)}')
            
    return interpolation_err

def avg_flow_interpolate(all_frame_pts, frame_idx=None, window_size=2, center_pos=None, non_missing_mask=None, align=False, verbose=True):
    """
    Interpolate the points in all_frame_pts using the average displacement method.

    Args:
        all_frame_pts (np.ndarray): The points to interpolate. Shape: (frame_cnt, pt_cnt, 2).
        frame_idx (int): The index of the frame to interpolate.
        window_size (int, optional): The size of the window to use for interpolation (inlusive on both ends). Defaults to 2.
        center_pos (tuple, optional): The center position of the points. Defaults to None.

    Returns:
        np.ndarray: The interpolated points at frame_idx. Shape: (pt_cnt, 2).
    """
    if frame_idx is not None:
        return avg_flow_interpolate_single_frame(all_frame_pts, frame_idx, window_size=window_size, center_pos=center_pos, non_missing_mask=non_missing_mask, align=align, verbose=verbose)
    else:
        frame_cnt = all_frame_pts.shape[0]
        interpolated_pts = np.zeros_like(all_frame_pts)
        for frame_idx in range(frame_cnt):
            interpolated_pts[frame_idx] = avg_flow_interpolate_single_frame(all_frame_pts, frame_idx, window_size=window_size, center_pos=center_pos, non_missing_mask=non_missing_mask, align=align, verbose=verbose)
        return interpolated_pts
    
def avg_flow_interpolate_single_frame(all_frame_pts, frame_idx, window_size=2, center_pos=None, non_missing_mask=None, align=False, verbose=True):
    if center_pos is None:
        center_pos = all_frame_pts.mean(axis=0)
    if frame_idx < window_size:
        return polar_interpolate(all_frame_pts[frame_idx], center_pos=center_pos, verbose=verbose)
    
    pt_cnt = all_frame_pts.shape[1]
    curr_frame_pts = all_frame_pts[frame_idx]
    prev_frame_pts = all_frame_pts[frame_idx - 1]
    avg_flow_vecs = (all_frame_pts[frame_idx - 1] - all_frame_pts[frame_idx - window_size]) / (window_size - 1)
    if non_missing_mask is None:
        non_missing_mask = get_missing_count(curr_frame_pts) == 0
        first_non_missing_idx = np.where(non_missing_mask)[0][0]
        non_missing_mask = non_missing_mask[frame_idx]
    else:
        if non_missing_mask.ndim == 2:
            non_missing_mask = non_missing_mask[frame_idx]
        first_non_missing_idx = np.where(non_missing_mask)[0]
    
    missing_count = (~non_missing_mask).sum()
    if missing_count > 0 and verbose:
        print(f'missing count = {missing_count} at frame {frame_idx}')
    
    curr_frame_pts = np.roll(curr_frame_pts, -first_non_missing_idx, axis=0)
    prev_frame_pts = np.roll(prev_frame_pts, -first_non_missing_idx, axis=0)
    non_missing_mask = np.roll(non_missing_mask, -first_non_missing_idx, axis=0)
    avg_flow_vecs = np.roll(avg_flow_vecs, -first_non_missing_idx, axis=0)
    non_missing_indices = np.where(non_missing_mask)[0]
    
    for pt_idx in range(pt_cnt):
        if non_missing_mask[pt_idx]:
            continue
        prev_idx = non_missing_indices[non_missing_indices < pt_idx][-1] if any(non_missing_indices < pt_idx) else non_missing_indices[-1]
        next_idx = non_missing_indices[non_missing_indices > pt_idx][0] if any(non_missing_indices > pt_idx) else non_missing_indices[0]
        
        if next_idx > prev_idx:
            denom = next_idx - prev_idx
        else:
            denom = pt_cnt - (prev_idx - next_idx)
        weight = ((pt_idx - prev_idx) % pt_cnt) / denom
        avg_flow = avg_flow_vecs[prev_idx] * weight + avg_flow_vecs[next_idx] * (1 - weight)
        curr_frame_pts[pt_idx] = prev_frame_pts[pt_idx] + avg_flow
    
    curr_frame_pts = np.roll(curr_frame_pts, first_non_missing_idx, axis=0)
    return curr_frame_pts

def test_single_frame_avg_flow_interpolation(gt_pts, frame_idx, window_size=2, reorder=True):
    gt_pts_copy = gt_pts.copy()
    curr_frame_pts = gt_pts[frame_idx]
    if reorder:
        poly_order = poly_4(curr_frame_pts)
        curr_frame_pts = curr_frame_pts[poly_order]
        gt_pts_copy = gt_pts_copy[:, poly_order]
    interp_pts = np.zeros_like(curr_frame_pts)
    
    for pt_idx in range(curr_frame_pts.shape[0]):
        curr_pt = gt_pts_copy[frame_idx, pt_idx].copy()
        gt_pts_copy[frame_idx, pt_idx] = (0, 0)
        interp_pts[pt_idx] = avg_flow_interpolate(gt_pts_copy, frame_idx, window_size=window_size, verbose=False)[pt_idx]
        gt_pts_copy[frame_idx, pt_idx] = curr_pt
    plot_pred_vs_gt_pts(interp_pts, curr_frame_pts)
    mean_err = np.linalg.norm(interp_pts - curr_frame_pts, axis=1).mean()
    print(f'mean error (pixel): {mean_err:.2f}')