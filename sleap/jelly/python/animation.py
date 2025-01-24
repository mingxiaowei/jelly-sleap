import numpy as np
import sleap
import matplotlib.pyplot as plt
from matplotlib import animation

def get_next_frame_idx(all_tracked_points: np.array, frame_idx: int, inst_idx: int) -> int:
    """
    Get the index of the next frame with tracked points.
    """
    frame_cnt = all_tracked_points.shape[0]
    next_frame_idx = frame_idx + 1
    while next_frame_idx < frame_cnt and all_tracked_points[next_frame_idx, inst_idx].sum() == 0:
        next_frame_idx += 1
    return max(next_frame_idx, frame_cnt - 1)

def get_prev_frame_idx(all_tracked_points: np.array, frame_idx: int, inst_idx: int) -> int:
    """
    Get the index of the previous frame with tracked points.
    """
    prev_frame_idx = frame_idx - 1
    while prev_frame_idx >= 0 and all_tracked_points[prev_frame_idx, inst_idx].sum() == 0:
        prev_frame_idx -= 1
    return max(prev_frame_idx, 0)

def get_all_tracked_points(label: sleap.Labels, reorder: bool = True) -> np.ndarray:
    """
    Get all tracked points from a sleap label file.
    """
    frame_cnt = len(label.labeled_frames)
    instance_cnt = len(label.labeled_frames[0].instances)
    all_tracked_points = np.zeros((frame_cnt, instance_cnt, 2))

    # populate all_tracked_coords with known coordinates
    for lf in label.labeled_frames:
        for instance in lf.instances:
            track_idx = int(instance.track.name.split('_')[-1])
            all_tracked_points[lf.frame_idx, track_idx] = instance.points_array[0]

    # interpolate missing coordinates
    missing_point_cnt = 0
    for frame_idx in range(frame_cnt):
        for inst_idx in range(instance_cnt):
            if all_tracked_points[frame_idx, inst_idx].sum() == 0:
                missing_point_cnt += 1
                prev_frame_idx = get_prev_frame_idx(all_tracked_points, frame_idx, inst_idx)
                next_frame_idx = get_next_frame_idx(all_tracked_points, frame_idx, inst_idx)
                all_tracked_points[frame_idx, inst_idx] = (all_tracked_points[prev_frame_idx, inst_idx] + all_tracked_points[next_frame_idx, inst_idx]) / 2
    print(f"Missing point count: {missing_point_cnt}")
    
    if reorder:
        reorder_idx = find_polygon_order(all_tracked_points[0])
        all_tracked_points = all_tracked_points[:, reorder_idx, :]
    
    return all_tracked_points

def find_polygon_order(points: np.ndarray) -> np.ndarray:
    """
    Find order of points to form a polygon using nearest neighbor algorithm
    
    Args:
        points: (N,2) array of point coordinates
        
    Returns:
        order: array of indices giving the order to connect points
    """
    N = len(points)
    unvisited = set(range(N))
    
    # Start with point closest to top of image
    current = min(range(N), key=lambda i: points[i,1]) 
    
    order = [current]
    unvisited.remove(current)
    
    # Add closest unvisited point at each step
    while unvisited:
        current = min(unvisited, 
                     key=lambda i: ((points[i,0] - points[current,0])**2 + 
                                  (points[i,1] - points[current,1])**2))
        order.append(current)
        unvisited.remove(current)
        
    return np.array(order)

def get_animation(
        label: sleap.Labels,
        reorder: bool = True,
        fps: int = 50,
        output_path: str = None
    ) -> animation.FuncAnimation:
    """
    Get an animation of the tracked points.
    """
    frame_cnt, x, y = label.video.shape[:3]
    all_tracked_points = get_all_tracked_points(label, reorder)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, x)
    ax.set_ylim(0, y)
    ax.invert_yaxis()  # Invert y-axis since image coordinates start from top

    # Initialize empty line and scatter objects
    line, = ax.plot([], [], 'b-', lw=1)  # Line for edges
    scat = ax.scatter([], [], c='red', s=30)  # Points

    def init():
        line.set_data([], [])
        scat.set_offsets(np.zeros((0, 2)))  # Correct way to initialize empty scatter
        return line, scat

    def animate(frame):
        # Get points for current frame
        points = all_tracked_points[frame]  # Shape: (17, 2)
        
        # Add first point to end to close the polygon
        points_closed = np.vstack([points, points[0]])
        
        # Update line (edges)
        line.set_data(points_closed[:, 0], points_closed[:, 1])
        
        # Update scatter (points)
        scat.set_offsets(points)
        
        return line, scat

    # Create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frame_cnt, interval=50, blit=True)

    # Optional: save animation
    if output_path is not None:
        anim.save(output_path, writer='ffmpeg', fps=fps)
        print(f"Animation saved to {output_path}")
    return anim
