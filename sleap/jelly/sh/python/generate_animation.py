import os, sys
sys.path.append('/home/mingxiao/Desktop/jelly-sleap/sleap/jelly/')

from eval.eval import *
from autoencoder.src.data_loader import *

results_affix = ['m11t1', 'm11t2', 'm11t3', 'm11t4', 'm11t5']
all_predicted_pointss = []
results_dir = '/home/mingxiao/Desktop/jelly-sleap/sleap/jelly/autoencoder/results/r7'
existing_results = os.listdir(results_dir)
for affix in results_affix:
    print(f'affix: {affix}')
    denoised_coords = np.load(os.path.join(results_dir, f'prediction_{affix}_corrected.npy'))
    print(denoised_coords.shape)
    all_predicted_pointss.append(denoised_coords)

corrected_dataset_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/correction_test/a1_1h_20s_corrected_with_scores_simplemax.slp'
corrected_dataset = sleap.load_file(corrected_dataset_path)
print(corrected_dataset)
gt_pts = get_all_tracked_points(corrected_dataset, interpolate=False, use_labeled_only=False, reorder=False, start_idx=0)

vid_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/animal_1_1h_20s.mp4'
vid = video_loader(vid_path)

anim_path = '/home/mingxiao/Desktop/animation/ae_r7_m11_eval_v2.mp4'
pred_simplemax_pts = np.load('/home/mingxiao/Desktop/jellyfish/video/video_1_clips/correction_test/a1_1h_20s_pred_simplemax_pts_avg_flow_interpolated.npy')
all_predicted_pts = all_predicted_pointss
poly_order = poly_4(pred_simplemax_pts[0])
pred_simplemax_pts = pred_simplemax_pts[:, poly_order]
frames = 3000
pts_lst = [pred_simplemax_pts[:frames]] + [pp[:frames] for pp in all_predicted_pts]
label_lst = ['before'] + results_affix
animate_tb(pts_lst, label_lst=label_lst, bg_video=vid[:frames], bg_video_start_idx=0, output_path=anim_path, 
           fps=10, subplot_shape=(2, 3))
