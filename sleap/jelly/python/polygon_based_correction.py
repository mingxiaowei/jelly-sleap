import numpy as np
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.heuristics import solve_tsp_local_search

import sys
sys.path.append('/home/mingxiao/Desktop/jelly-sleap/sleap/jelly/python/')
from postprocess import *

def find_best_roll(curr_frame_pts: np.ndarray, prev_frame_pts: np.ndarray) -> int:
    n = len(curr_frame_pts)
    comparator = lambda n: np.sum((np.roll(curr_frame_pts, n, axis=0) - prev_frame_pts) ** 2)
    return min(range(n), key=comparator)

def get_corrected_coords(all_tracked_points, id_mapping):
    corrected_coords = all_tracked_points.copy()
    for frame_idx in range(1, all_tracked_points.shape[0]):
        corrected_coords[frame_idx] = all_tracked_points[frame_idx][id_mapping[frame_idx]]
    return corrected_coords

def poly_3(all_tracked_points):
    points = all_tracked_points
    if len(points) <= 2:
        return list(range(len(points)))
    initial_permutation = nearest_neighbor_path(points)
    distance_matrix = euclidean_distance_matrix(points)
    permutation, _ = solve_tsp_local_search(distance_matrix, x0=initial_permutation)
    return permutation

def polygon_correction(labels, mins_score=0.4, polygon_constructor=poly_3):
    all_tracked_points = get_all_tracked_points(labels, reorder=True, min_score=mins_score)
    return polygon_correction_with_points(all_tracked_points, polygon_constructor)

def polygon_correction_with_points(all_tracked_points, polygon_constructor=poly_3, return_indices=False):
    id_mapping = get_id_mapping_array(all_tracked_points)
    id_mapping = id_mapping.copy()
    frame_cnt = all_tracked_points.shape[0]
    idx_range = np.arange(all_tracked_points.shape[1])

    for frame_idx in range(1, frame_cnt):
        curr_frame_pts = all_tracked_points[frame_idx]
        polygon_order = polygon_constructor(curr_frame_pts)
        prev_frame_indices = id_mapping[frame_idx-1]
        best_shift = find_best_roll(curr_frame_pts[polygon_order], all_tracked_points[frame_idx-1][prev_frame_indices])
        shifted_indices = np.roll(polygon_order, best_shift, axis=0)
        id_mapping[frame_idx] = idx_range[shifted_indices]

    corrected_coords = get_corrected_coords(all_tracked_points, id_mapping)
    if return_indices:
        return corrected_coords, id_mapping
    else:
        return corrected_coords

def poly_1(points):
    center_pos = np.mean(points, axis=0)
    coord_diff = points - center_pos
    angles = np.arctan2(coord_diff[:, 1], coord_diff[:, 0])
    sorted_indices = np.argsort(angles)
    return sorted_indices

def compute_total_length(points, path):
    total = 0.0
    N = len(path)
    for i in range(N):
        a = path[i]
        b = path[(i + 1) % N]
        total += np.linalg.norm(points[a] - points[b])
    return total

def nearest_neighbor_path(points):
    N = points.shape[0]
    if N == 0:
        return []
    start = np.lexsort((points[:, 1], points[:, 0]))[0]
    path = [start]
    visited = set([start])
    for _ in range(N - 1):
        current = path[-1]
        min_dist = np.inf
        next_point = -1
        for i in range(N):
            if i not in visited:
                dist = np.linalg.norm(points[current] - points[i])
                if dist < min_dist:
                    min_dist = dist
                    next_point = i
        path.append(next_point)
        visited.add(next_point)
    return path

def two_opt(points, path):
    path = path.copy()
    N = len(path)
    best_length = compute_total_length(points, path)
    improved = True
    while improved:
        improved = False
        for i in range(N):
            for j in range(i + 1, N):
                new_path = path.copy()
                new_path[i + 1:j + 1] = new_path[j:i:-1]
                new_length = compute_total_length(points, new_path)
                if new_length < best_length:
                    path = new_path
                    best_length = new_length
                    improved = True
                    break
            if improved:
                break
    return path

def poly_2(all_tracked_points):
    points = all_tracked_points
    N = points.shape[0]
    if N <= 2:
        return list(range(N))
    initial_path = nearest_neighbor_path(points)
    optimized_path = two_opt(points, initial_path)
    return optimized_path

def get_frame_missing_cnt_arr(all_tracked_points):
    frame_cnt, inst_cnt = all_tracked_points.shape[:2]
    frame_missing_cnt = np.zeros(frame_cnt)
    for frame_idx in range(frame_cnt):
        for inst_idx in range(inst_cnt):
            if all_tracked_points[frame_idx, inst_idx].sum() == 0:
                frame_missing_cnt[frame_idx] += 1
    return frame_missing_cnt

def make_ccw(points, permutation=None, polygon_constructor=poly_3):
    if permutation is None:
        permutation = polygon_constructor(points)
    x = points[permutation, 0]
    y = points[permutation, 1]
    area = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) + 0.5 * (x[-1] * y[0] - x[0] * y[-1])
    
    # If area is positive, the polygon is counterclockwise, so reverse the permutation
    if area < 0:
        permutation = permutation[::-1]
    
    return permutation

def poly_4(points):
    return make_ccw(points, polygon_constructor=poly_3)

def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi

# adapted from https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
def cart2pol(xy_arr, center_pos=(0, 0)):
    xy_arr = xy_arr - center_pos
    x = xy_arr[:, 0]
    y = xy_arr[:, 1]
    rho = np.hypot(x, y)  # hypot always returns non-negative values
    phi = np.arctan2(y, x)
    # Convert phi from [-π, π] to [0, 2π] range
    phi = -phi + np.pi
    return np.array([rho, phi]).T

def pol2cart(rho_phi_arr, center_pos=(0, 0)):
    rho = np.abs(rho_phi_arr[:, 0])  # Ensure rho is positive
    phi = np.mod(rho_phi_arr[:, 1], 2*np.pi)  # Ensure phi is in [0, 2π]
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y]).T + center_pos

def polar_interpolate(curr_frame_pts, non_missing_mask=None, center_pos=None):
    
    if non_missing_mask is None:
        non_missing_mask = ~(curr_frame_pts == 0).all(axis=1)
    
    first_non_missing_idx = np.where(non_missing_mask)[0][0]
    curr_frame_pts = np.roll(curr_frame_pts, -first_non_missing_idx, axis=0)
    non_missing_mask = np.roll(non_missing_mask, -first_non_missing_idx, axis=0)
    non_missing_indices = np.where(non_missing_mask)[0]
    
    if center_pos is None:
        center_pos = curr_frame_pts[non_missing_mask].mean(axis=0)
    
    polar_coords = cart2pol(curr_frame_pts, center_pos)
    cart_interpolated_pts = np.zeros_like(curr_frame_pts)
    pt_cnt = len(curr_frame_pts)
    
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
        r1, theta1 = polar_coords[prev_idx]
        r2, theta2 = polar_coords[next_idx]

        r_interp = r1 + abs(weight) * (r2 - r1)
        if theta1 > theta2:
            theta2 += 2*np.pi
        theta_interp = np.mod(theta1 + weight * (theta2 - theta1), 2*np.pi)
        
        cart_interpolated_pts[pt_idx] = pol2cart(np.array([[r_interp, theta_interp]]), center_pos)[0]

    cart_interpolated_pts = np.roll(cart_interpolated_pts, first_non_missing_idx, axis=0)
    return cart_interpolated_pts
