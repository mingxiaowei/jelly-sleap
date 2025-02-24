import sleap
import sys
import cmath

sys.path.append('..')
sys.path.append('../..')
from python.polygon_based_correction import *
from eval.eval import *

if __name__ == '__main__':

    single_model_points_corrected_path = '/home/mingxiao/Desktop/jellyfish/label/predictions/single_model_all_tracked_points_corrected.npy'
    single_model_points_corrected = np.load(single_model_points_corrected_path)
    print(single_model_points_corrected.shape)

    single_model_points_corrected_eval_path = '/home/mingxiao/Desktop/jellyfish/label/predictions/single_model_all_tracked_points_corrected_eval.npy'
    eval_dataset_from_points(single_model_points_corrected, method='polygon', save_path=single_model_points_corrected_eval_path)