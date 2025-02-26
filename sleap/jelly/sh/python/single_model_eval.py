import sleap
import sys
import cmath

sys.path.append('..')
sys.path.append('../..')
from python.polygon_based_correction import *
from eval.eval import *

if __name__ == '__main__':

    # single_model_points_corrected_path = '/home/mingxiao/Desktop/jellyfish/label/predictions/single_model_all_tracked_points_corrected.npy'
    single_model_points_path = '/home/mingxiao/Desktop/jellyfish/label/predictions/single_model_all_tracked_points.npy'
    single_model_points_uncorrected = np.load(single_model_points_path)
    print(single_model_points_uncorrected.shape)

    # single_model_points_corrected_eval_path = '/home/mingxiao/Desktop/jellyfish/label/predictions/single_model_all_tracked_points_corrected_eval.npy'
    swap_cnt_lst, missing_cnt_lst,filtered_ranges = eval_dataset_from_points(single_model_points_uncorrected, method='polygon')
    save_path = '/home/mingxiao/Desktop/jellyfish/label/predictions/single_model_uncorrected_eval_'
    np.save(f'{save_path}_swap_cnt_lst.npy', swap_cnt_lst)
    np.save(f'{save_path}_missing_cnt_lst.npy', missing_cnt_lst)
    np.save(f'{save_path}_filtered_ranges.npy', filtered_ranges)