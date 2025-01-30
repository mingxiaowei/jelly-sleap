import sleap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .animation import *

def get_all_radii(tracked_points, center_pos=np.array([91, 77])):
    all_radii = np.zeros(tracked_points.shape[:2])
    for i in range(tracked_points.shape[0]):
        for j in range(tracked_points.shape[1]):
            all_radii[i, j] = np.linalg.norm(tracked_points[i, j] - center_pos)
    return all_radii

def plot_radii_distribution(all_radii):
    plt.figure(figsize=(10, 6))

    # Plot KDE for each point
    for i in range(all_radii.shape[1]):
        sns.kdeplot(data=all_radii[:, i], label=f'Point {i}')

    plt.xlabel('Radius')
    plt.ylabel('Density')
    plt.title('Distribution of Radii for Each Point')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_radii_over_time(all_radii, first_x_proportion=1):
    assert 0 < first_x_proportion <= 1, "first_x_proportion must be between 0 and 1"
    plt.figure(figsize=(12, 6))
    first_x_radii = all_radii[:int(all_radii.shape[0] * first_x_proportion)]

    # Plot time series for each point
    for i in range(first_x_radii.shape[1]):
        plt.plot(range(first_x_radii.shape[0]), first_x_radii[:, i], label=f'Point {i}', alpha=0.7)

    plt.xlabel('Frame Number')
    plt.ylabel('Radius (pixels)')
    plt.title(f'Radius Over Time for Each Point (first {int(first_x_proportion * 100)}% of total frames)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_radii_derivative(all_radii, first_x_proportion=1):
    first_x_radii = all_radii[:int(all_radii.shape[0] * first_x_proportion)]
    plt.figure(figsize=(12, 6))

    # Calculate derivatives (differences between consecutive frames)
    derivatives = np.diff(first_x_radii, axis=0)

    # Plot derivatives for each point
    for i in range(derivatives.shape[1]):
        plt.plot(range(derivatives.shape[0]), derivatives[:, i], label=f'Point {i}', alpha=0.7)

    plt.xlabel('Frame Number') 
    plt.ylabel('Change in Radius (pixels/frame)')
    plt.title(f'Rate of Change in Radius Over Time for Each Point (first {int(first_x_proportion * 100)}% of total frames)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def get_expanded_periods(all_radii, min_range_length=100, mean_scale=0.8, derivative_thres=5, plot=True):
    plt.figure(figsize=(12, 6))

    # Calculate derivatives
    derivatives = np.diff(all_radii, axis=0)

    # Find ranges where all derivatives are between -5 and 5
    all_stable = np.all((derivatives > -derivative_thres) & (derivatives < derivative_thres), axis=1)
    ranges = []
    start_idx = None

    # Find continuous ranges
    for i in range(len(all_stable)):
        if all_stable[i] and start_idx is None:
            start_idx = i
        elif not all_stable[i] and start_idx is not None:
            if i - start_idx >= min_range_length:  # Only keep ranges at least 100 frames long
                ranges.append((start_idx, i))
            start_idx = None
            
    if start_idx is not None and len(all_stable) - start_idx >= min_range_length:
        ranges.append((start_idx, len(all_stable)))

    # Calculate mean radius for each point
    mean_radii = np.mean(all_radii, axis=0) * mean_scale
    print(f'mean_radii: {mean_radii}')
    # Filter ranges to only include frames where all radii are above their means
    filtered_ranges = []
    for start, end in ranges:
        # For each frame in range, check if all points are above their means
        above_mean_mask = np.all(all_radii[start:end] > mean_radii, axis=1)
        
        # Find continuous sub-ranges where all points are above mean
        sub_start_idx = None
        for i in range(len(above_mean_mask)):
            if above_mean_mask[i] and sub_start_idx is None:
                sub_start_idx = start + i
            elif (not above_mean_mask[i] or i == len(above_mean_mask)-1) and sub_start_idx is not None:
                sub_end_idx = start + i
                if sub_end_idx - sub_start_idx >= min_range_length:
                    filtered_ranges.append((sub_start_idx, sub_end_idx))
                sub_start_idx = None
    filtered_ranges = np.array(filtered_ranges)

    # Print the filtered ranges
    print(f"Stable ranges (frame numbers) with minimum length of {min_range_length} frames and all radii above {mean_scale*100}% mean:")
    for start, end in filtered_ranges:
        print(f"    - Frames {start} to {end} (length: {end-start})")

    if plot:
        # Plot derivatives for each point
        for i in range(derivatives.shape[1]):
            plt.plot(range(derivatives.shape[0]), derivatives[:, i], label=f'Point {i}', alpha=0.7)

        # Add vertical bars for filtered ranges
        for start, end in filtered_ranges:
            plt.axvspan(start, end, color='green', alpha=0.1)

        plt.xlabel('Frame Number')
        plt.ylabel('Change in Radius (pixels/frame)')
        plt.title(f'Rate of Change in Radius Over Time\nGreen regions: stable periods with all radii above {mean_scale*100}% mean')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return filtered_ranges

def get_id_mapping_array(all_tracked_points: np.ndarray) -> np.ndarray:
    """
    Initialize the id mapping array for the tracked points
    """
    frame_cnt, pt_cnt = all_tracked_points.shape[:2]
    return np.tile(np.arange(pt_cnt), (frame_cnt, 1))