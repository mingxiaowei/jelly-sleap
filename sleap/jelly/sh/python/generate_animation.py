import os, sys
sys.path.append('/home/mingxiao/Desktop/jelly-sleap/sleap/jelly/')

from eval.eval import *
from autoencoder.src.data_loader import *

thresholds = [4, 6, 8, 10, 12]
all_pp3_pts_3k = []
pp_dir = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/postprocess'
for thres in thresholds:
    pp_path = os.path.join(pp_dir, f'a1_raw_pts_pp3_thres{thres}.npy')
    pp_pts = np.load(pp_path)
    all_pp3_pts_3k.append(pp_pts)
all_pp3_pts_3k = np.array(all_pp3_pts_3k)
print(all_pp3_pts_3k.shape)



vid_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/animal_1_1h_20s.mp4'
vid = video_loader(vid_path)

input_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/correction_test/a1_1h_20s_pred_simplemax_pts_preprocessed_v6.npy'
input_pts = np.load(input_path)

anim_path = '/home/mingxiao/Desktop/animation/a1_pp_eval.mp4'
pred_simplemax_pts = np.load('/home/mingxiao/Desktop/jellyfish/video/video_1_clips/correction_test/a1_1h_20s_pred_simplemax_pts.npy')
poly_order = poly_4(pred_simplemax_pts[0])
pred_simplemax_pts = pred_simplemax_pts[:, poly_order]

frames = 3000
pts_lst = [pred_simplemax_pts[:frames]] + [input_pts[:frames]] + [pp[:frames] for pp in all_pp3_pts_3k]
label_lst = ['original'] + ['input'] + [f'thres={thres}' for thres in thresholds]

animate_tb(pts_lst, label_lst=label_lst, bg_video=vid[:frames], bg_video_start_idx=0, output_path=anim_path, 
           fps=10, subplot_shape=(2, 4))
