import sys 
sys.path.append('../..')
sys.path.append('/home/mingxiao/Desktop/jelly-sleap/sleap/jelly')
from python.polygon_based_correction import *
from eval.eval import *

from tqdm import tqdm

single_model_points_path = '/home/mingxiao/Desktop/jellyfish/label/predictions/single_model_all_tracked_points.npy'
single_model_points = np.load(single_model_points_path)

def assign_id_to_points(all_tracked_points, frame_idx, polygon_constructor=poly_4):
    curr_frame_pts = all_tracked_points[frame_idx].copy()
    pt_cnt = len(curr_frame_pts)
    non_missing_mask = np.isnan(curr_frame_pts).sum(axis=1) == 0 # 1 = non-missing, 0 = missing
    
    # for missing points, interpolate from its two neighbors using polar coordinates
    polar_interpolated_pts = polar_interpolate(all_tracked_points[frame_idx], non_missing_mask)
    for pt_idx in range(pt_cnt):
        if non_missing_mask[pt_idx]:
            continue
        all_tracked_points[frame_idx][pt_idx] = polar_interpolated_pts[pt_idx]
    
    # assign order again to ensure maximal global alignment
    assign_id_to_points_non_missing(all_tracked_points, frame_idx, polygon_constructor)
        
def assign_id_to_points_non_missing(all_tracked_points, frame_idx, polygon_constructor=poly_4):
    curr_frame_pts = all_tracked_points[frame_idx]
    polygon_order = polygon_constructor(curr_frame_pts)
    best_shift = find_best_roll(curr_frame_pts[polygon_order], all_tracked_points[frame_idx-1])
    shifted_indices = np.roll(polygon_order, best_shift, axis=0)
    all_tracked_points[frame_idx] = curr_frame_pts[shifted_indices]

def polygon_correction_with_points(all_tracked_points, polygon_constructor=poly_4, return_indices=False):

    frame_cnt = all_tracked_points.shape[0]
    corrected_coords = all_tracked_points.copy()

    for frame_idx in tqdm(range(1, frame_cnt)):
        assign_id_to_points(corrected_coords, frame_idx, polygon_constructor)

    return corrected_coords

single_model_points_corrected = polygon_correction_with_points(single_model_points)
single_model_points_corrected_path = '/home/mingxiao/Desktop/jellyfish/label/predictions/single_model_all_tracked_points_corrected.npy'
np.save(single_model_points_corrected_path, single_model_points_corrected)