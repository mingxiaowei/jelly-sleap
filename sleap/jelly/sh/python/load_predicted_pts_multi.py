import sys 
sys.path.append('../..')
sys.path.append('/home/mingxiao/Desktop/jelly-sleap/sleap/jelly/')
from python.animation import *

multi_4k_dataset_path = '/home/mingxiao/Desktop/jellyfish/label/multifish/predictions/multifish_animal_1_v10.slp.250221_051111.predictions.slp'
multi_4k_dataset = sleap.load_file(multi_4k_dataset_path)

multi_4k_tracked_points = get_all_untracked_points(multi_4k_dataset, 
                                                   reorder=True, 
                                                   interpolate=False, 
                                                   min_score=0.5, 
                                                   start_idx=0, 
                                                   use_labeled_only=False,
                                                   tb_cnt=17)
multi_4k_points_path = '/home/mingxiao/Desktop/jellyfish/label/multifish/predictions/multifish_animal_1_v10_predicted_points.npy'
np.save(multi_4k_points_path, multi_4k_tracked_points)