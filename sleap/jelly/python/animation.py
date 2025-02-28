import numpy as np
import sleap
import os
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Union, List
from .polygon_based_correction import poly_4
from tqdm import tqdm

def get_next_frame_idx(all_tracked_points: np.array, frame_idx: int, inst_idx: int) -> int:
    """
    Get the index of the next frame with tracked points.
    """
    frame_cnt = all_tracked_points.shape[0]
    next_frame_idx = frame_idx + 1
    while next_frame_idx < frame_cnt and all_tracked_points[next_frame_idx, inst_idx].sum() == 0:
        next_frame_idx += 1
    return min(next_frame_idx, frame_cnt - 1)

def get_prev_frame_idx(all_tracked_points: np.array, frame_idx: int, inst_idx: int) -> int:
    """
    Get the index of the previous frame with tracked points.
    """
    prev_frame_idx = frame_idx - 1
    while prev_frame_idx >= 0 and all_tracked_points[prev_frame_idx, inst_idx].sum() == 0:
        prev_frame_idx -= 1
    return max(prev_frame_idx, 0)

def get_all_tracked_points(label: sleap.Labels, 
                           reorder: bool = True, 
                           interpolate: bool = True,
                           min_score: int = 0, 
                           start_idx: int = 1) -> np.ndarray:
    """
    Get all tracked points from a sleap label file.
    """
    labeled_frames_to_use = label.labeled_frames[start_idx:]
    frame_cnt = len(labeled_frames_to_use)
    instance_cnt = len(labeled_frames_to_use[0].instances)
    all_tracked_points = np.zeros((frame_cnt, instance_cnt, 2))
    print(f'all_tracked_points shape: {all_tracked_points.shape}')

    # populate all_tracked_coords with known coordinates
    for lf in labeled_frames_to_use:
        for instance in lf.instances:
            if isinstance(instance, sleap.instance.PredictedInstance) and instance.score < min_score:
                continue
            track_idx = int(instance.track.name.split('_')[-1])
            all_tracked_points[lf.frame_idx - start_idx, track_idx] = instance.points_array[0]

    # interpolate missing coordinates
    missing_point_cnt = 0
    first_non_missing_frame_idx = None
    for frame_idx in range(frame_cnt):
        curr_frame_missing_point_cnt = 0
        for inst_idx in range(instance_cnt):
            if all_tracked_points[frame_idx, inst_idx].sum() == 0:
                curr_frame_missing_point_cnt += 1
                if interpolate:
                    prev_frame_idx = get_prev_frame_idx(all_tracked_points, frame_idx, inst_idx)
                    next_frame_idx = get_next_frame_idx(all_tracked_points, frame_idx, inst_idx)
                    all_tracked_points[frame_idx, inst_idx] = (all_tracked_points[prev_frame_idx, inst_idx] + all_tracked_points[next_frame_idx, inst_idx]) / 2
        missing_point_cnt += curr_frame_missing_point_cnt
        if first_non_missing_frame_idx is None and curr_frame_missing_point_cnt == 0:
            first_non_missing_frame_idx = frame_idx
            print(f"First non missing frame idx: {first_non_missing_frame_idx}")
    print(f"Missing point count: {missing_point_cnt}")
    
    if reorder: 
        reorder_idx = poly_4(all_tracked_points[first_non_missing_frame_idx])
        print(f"reorder_idx.shape: {reorder_idx.shape}")
        all_tracked_points = all_tracked_points[:, reorder_idx, :]
    
    return all_tracked_points

def get_all_untracked_points(label: sleap.Labels, 
                           reorder: bool = True, 
                           interpolate: bool = True,
                           min_score: int = 0, 
                           use_labeled_only: bool = True,
                           start_idx: int = 0, 
                           tb_cnt: int = 17) -> np.ndarray:
    """
    Get all untracked points from a sleap label file.
    """
    labeled_frames_to_use = label.labeled_frames[start_idx:]
    return get_all_untracked_points_from_lbfs(labeled_frames_to_use, reorder, interpolate, min_score, use_labeled_only, start_idx, tb_cnt)

def get_all_untracked_points_from_lbfs(lbfs: List[sleap.instance.LabeledFrame],
                                       reorder: bool = True, 
                                       interpolate: bool = True,
                                       min_score: float = 0,
                                       use_labeled_only: bool = True,
                                       start_idx: int = 0, 
                                       tb_cnt: int = 17) -> np.ndarray:
    """
    Get all untracked points from a list of labeled frames.
    """
    frame_cnt = len(lbfs)
    # instance_cnt = len(lbfs[0].instances)
    instance_cnt = tb_cnt
    all_untracked_points = np.zeros((frame_cnt, instance_cnt, 2))
    print(f'all_untracked_points shape: {all_untracked_points.shape}')
    missing_point_cnt = 0

    # populate all_tracked_coords with known coordinates
    if use_labeled_only:
        validator = lambda inst: isinstance(inst, sleap.instance.Instance) and not isinstance(inst, sleap.instance.PredictedInstance)
    else:
        validator = lambda inst: isinstance(inst, sleap.instance.PredictedInstance) and inst.score >= min_score
        
    for frame_idx, lf in tqdm(enumerate(lbfs)):
        pred_insts = [instance for instance in lf.instances if validator(instance)]
        if len(pred_insts) > tb_cnt:
            pred_inst_scores = [instance.score for instance in pred_insts]
            sorted_args = np.argsort(pred_inst_scores)[::-1][:tb_cnt]
            pred_insts = [inst for i, inst in enumerate(pred_insts) if i in sorted_args]
        for inst_idx, instance in enumerate(pred_insts):
            pt_coord = instance.points_array[0]
            if pt_coord.sum() == 0:
                missing_point_cnt += 1
            all_untracked_points[frame_idx - start_idx, inst_idx] = pt_coord

    first_non_missing_frame_idx = start_idx
    # interpolate missing coordinates
    if interpolate:
        first_non_missing_frame_idx = None
        for frame_idx in range(frame_cnt):
            curr_frame_missing_point_cnt = 0
            for inst_idx in range(instance_cnt):
                if all_untracked_points[frame_idx, inst_idx].sum() == 0:
                    curr_frame_missing_point_cnt += 1
                    prev_frame_idx = get_prev_frame_idx(all_untracked_points, frame_idx, inst_idx)
                    next_frame_idx = get_next_frame_idx(all_untracked_points, frame_idx, inst_idx)
                    all_untracked_points[frame_idx, inst_idx] = (all_untracked_points[prev_frame_idx, inst_idx] + all_untracked_points[next_frame_idx, inst_idx]) / 2
            if first_non_missing_frame_idx is None and curr_frame_missing_point_cnt == 0:
                first_non_missing_frame_idx = frame_idx
                print(f"First non missing frame idx: {first_non_missing_frame_idx}")
    print(f"Missing point count: {missing_point_cnt}")
    
    if reorder: 
        reorder_idx = poly_4(all_untracked_points[first_non_missing_frame_idx])
        print(f"reorder_idx.shape: {np.array(reorder_idx).shape}")
        all_untracked_points = all_untracked_points[:, reorder_idx, :]
    
    return all_untracked_points

def get_all_tracked_points_single_model(label: sleap.Labels) -> np.ndarray:
    labeled_frames_to_use = label.labeled_frames
    frame_cnt = len(labeled_frames_to_use)
    instance_cnt = len(label.skeleton.nodes) - 1
    all_tracked_points = np.zeros((frame_cnt, instance_cnt, 2))
    print(f'all_tracked_points shape: {all_tracked_points.shape}')

    # populate all_tracked_coords with known coordinates
    for lf in labeled_frames_to_use:
            all_tracked_points[lf.frame_idx] = lf.instances[0].points_array[1:]
    
    return all_tracked_points

def find_polygon_order(points: np.ndarray) -> np.ndarray:
    """
    Find order of points to form a polygon by sorting based on angles from centroid
    
    Args:
        points: (N,2) array of point coordinates
        
    Returns:
        order: array of indices giving the order to connect points
    """
    N = len(points)
    if N < 3:
        return np.arange(N)
    
    # Calculate centroid
    centroid = points.mean(axis=0)
    
    # Calculate angles from centroid to all points
    angles = []
    for i in range(N):
        angle = np.arctan2(points[i,1] - centroid[1], points[i,0] - centroid[0])
        angles.append((angle, i))
    
    # Sort points by angle
    sorted_indices = [i for _, i in sorted(angles)]
    
    return np.array(sorted_indices)

def get_animation(
        label: sleap.Labels,
        reorder: bool = True,
        fps: int = 50,
        output_path: str = None
    ) -> animation.FuncAnimation:
    """
    Get an animation of the tracked points. 
    (First frame is skipped as there are one more instance than imposed max for some reason.) 
    """
    frame_cnt, x, y = label.video.shape[:3]
    frame_cnt -= 1  # Skip first frame 
    all_tracked_points = get_all_tracked_points(label, reorder)
    
    return get_animation_from_tracked_points(all_tracked_points, x, y, fps, output_path)

def get_animation_from_tracked_points(
        tracked_points_lst: Union[list, np.ndarray],  # List of tracked_points arrays
        x: int,
        y: int,
        fps: int = 50,
        bg_video: sleap.Video = None,
        bg_video_start_idx: int = 1,
        output_path: str = None
    ) -> animation.FuncAnimation:
    
    if not isinstance(tracked_points_lst, list):
        tracked_points_lst = [tracked_points_lst]
    frame_cnt = tracked_points_lst[0].shape[0]
    n_plots = len(tracked_points_lst)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_plots, figsize=(8*n_plots, 8))
    if n_plots == 1:
        axes = [axes]  # Make axes iterable when only one subplot
    
    # Initialize plot elements for each subplot
    lines = []
    scats = []
    bg_imgs = []
    
    for ax in axes:
        ax.set_xlim(0, x)
        ax.set_ylim(0, y)
        ax.invert_yaxis()
        
        # Initialize empty line and scatter objects for each subplot
        line, = ax.plot([], [], 'b-', lw=1)
        scat = ax.scatter([], [], c='red', s=30)
        bg_img = ax.imshow(np.zeros((y, x)), cmap='gray', vmin=0, vmax=255)
        
        lines.append(line)
        scats.append(scat)
        bg_imgs.append(bg_img)

    def init():
        plot_elements = []
        for line, scat, bg_img in zip(lines, scats, bg_imgs):
            line.set_data([], [])
            scat.set_offsets(np.zeros((0, 2)))
            plot_elements.extend([line, scat, bg_img])
        return plot_elements

    def animate(frame):
        plot_elements = []
        
        # Update background (same for all subplots)
        if bg_video is not None:
            bg_frame = bg_video.get_frame(frame + bg_video_start_idx)
            for bg_img in bg_imgs:
                bg_img.set_array(bg_frame[:, :, 0])
        
        # Update each subplot
        for i, (tracked_points, line, scat) in enumerate(zip(tracked_points_lst, lines, scats)):
            points = tracked_points[frame]
            points_closed = np.vstack([points, points[0]])
            
            line.set_data(points_closed[:, 0], points_closed[:, 1])
            scat.set_offsets(points)
            
            plot_elements.extend([line, scat, bg_imgs[i]])
            
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

def plot_polygon(points, order, ax, title):
    """Helper function to plot a polygon with given point order"""
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], c='blue', s=10)
    
    # Plot edges connecting points in order
    for i in range(len(order)):
        start = points[order[i]]
        end = points[order[(i + 1) % len(order)]]
        ax.plot([start[0], end[0]], [start[1], end[1]], 'r-', alpha=0.7)
        
        # Add point indices as labels
        ax.text(points[order[i], 0], points[order[i], 1], 
                str(order[i]), fontsize=12, ha='right')
    
    ax.set_title(title)
    ax.axis('equal')

def get_total_edge_length(points, order):
    total = 0
    for i in range(len(order)):
        start = points[order[i]]
        end = points[order[(i + 1) % len(order)]]
        total += np.sqrt(np.sum((end - start)**2))
    return total

def plot_some_polygons(points: np.ndarray, poly_constructors: list):
    poly_cnt = len(poly_constructors)
    polygons = [poly_constructor(points) for poly_constructor in poly_constructors]
    _, axs = plt.subplots(1, poly_cnt, figsize=(5*poly_cnt, 5))
    if poly_cnt == 1:
        axs = [axs]
    for i, order in enumerate(polygons):
        plot_polygon(points, order, axs[i], f'poly_{i+1}')
        axs[i].invert_yaxis() # invert y-axis to put (0,0) at top left
    plt.tight_layout()
    plt.show()
    
    print("\nTotal edge lengths:")
    for i, order in enumerate(polygons):
        print(f"poly_{i+1}: {get_total_edge_length(points, order):.3f}")