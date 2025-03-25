import os, sys
sys.path.append('/home/mingxiao/Desktop/jelly-sleap/sleap/jelly/')

from eval.eval import *
from autoencoder.src.data_loader import *

c1_vid_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_high_res_5min_track_reencoded_0.mp4'
c1_vid = video_loader(c1_vid_path)

pp_dir = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/postprocess'
c1_raw_pts_poly_corrected_path = '/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_raw_pts_poly_corrected.npy'
c1_pp_thres45_poly_corrected_path = os.path.join(pp_dir, 'c1_raw_pts_pp_thres15_v3.npy')
c1_pp3_thres45_poly_corrected_path = os.path.join(pp_dir, 'c1_raw_pts_pp3_thres15_v3.npy')
c1_raw_pts_poly_corrected = np.load(c1_raw_pts_poly_corrected_path)
c1_pp_thres45_poly_corrected = np.load(c1_pp_thres45_poly_corrected_path)
c1_pp3_thres45_poly_corrected = np.load(c1_pp3_thres45_poly_corrected_path)

frame_start = 1610
frame_end = 3000
anim_pts_lst = [c1_raw_pts_poly_corrected[frame_start:frame_end], c1_pp_thres45_poly_corrected[frame_start:frame_end], c1_pp3_thres45_poly_corrected[frame_start:frame_end]]
anim_label_lst = ['raw', 'avg flow (thres=15)', 'polar (thres=15)']
anim_path = '/home/mingxiao/Desktop/animation/c1_pp_v4.mp4'
animate_tb(anim_pts_lst, label_lst=anim_label_lst, output_path=anim_path, 
           bg_video=c1_vid[frame_start:frame_end], bg_video_start_idx=0, subplot_shape=(1, 3));