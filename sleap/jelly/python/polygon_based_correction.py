import numpy as np
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.heuristics import solve_tsp_local_search
from .animation import *
from .postprocess import *

def find_best_roll(curr_frame_indices: np.ndarray, prev_frame_indices: np.ndarray) -> int:
    """
    Find the optimal roll shift that maximizes alignment between two index arrays
    
    Args:
        curr_frame_indices: array of current frame indices
        prev_frame_indices: array of previous frame indices
        
    Returns:
        best_shift: integer shift that maximizes alignment
    """
    if np.array_equal(curr_frame_indices, prev_frame_indices):
        return 0
    n = len(curr_frame_indices)
    max_key = lambda n: np.sum(np.roll(curr_frame_indices, n) == prev_frame_indices)
    return max(range(n), key=max_key)

def get_corrected_coords(all_tracked_points, id_mapping):
    corrected_coords = all_tracked_points.copy()
    for frame_idx in range(1, all_tracked_points.shape[0]):
        corrected_coords[frame_idx] = all_tracked_points[frame_idx][id_mapping[frame_idx]]
    return corrected_coords

def polygon_correction(labels, polygon_constructor, mins_score=0.4):
    all_tracked_points = get_all_tracked_points(labels, reorder=True, min_score=mins_score)
    id_mapping = get_id_mapping_array(all_tracked_points)
    id_mapping = id_mapping.copy()
    frame_cnt = all_tracked_points.shape[0]
    idx_range = np.arange(all_tracked_points.shape[1])
    
    for frame_idx in range(1, frame_cnt):
        polygon_order = polygon_constructor(all_tracked_points[frame_idx])
        prev_frame_indices = id_mapping[frame_idx-1]
        best_shift = find_best_roll(polygon_order, prev_frame_indices)
        shifted_indices = np.roll(polygon_order, best_shift)        
        id_mapping[frame_idx] = idx_range[shifted_indices]
        
    corrected_coords = get_corrected_coords(all_tracked_points, id_mapping)
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

def poly_3(all_tracked_points):
    points = all_tracked_points
    if len(points) <= 2:
        return list(range(len(points)))
    initial_permutation = nearest_neighbor_path(points)
    distance_matrix = euclidean_distance_matrix(points)
    permutation, _ = solve_tsp_local_search(distance_matrix, x0=initial_permutation)
    return permutation
