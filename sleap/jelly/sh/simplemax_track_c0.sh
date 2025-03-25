#bin/bash
cd /home/mingxiao/Desktop/jellyfish/
PREDICTION_FOLDER="video/video_1_clips/predictions/"
OUTPUT_FOLDER="video/video_1_clips/predictions/flowmax-tracking/"

if [ ! -d "$OUTPUT_FOLDER" ]; then
    mkdir -p "$OUTPUT_FOLDER"
fi

n_threads=50
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads


PREDICTION_PATH='/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_raw_dataset_with_scores.slp'
OUTPUT_PATH='/home/mingxiao/Desktop/jellyfish/video/video_1_clips/c1_raw_dataset_simplemax.slp'

# micromamba deactivate
# micromamba activate sleap

sleap-track $PREDICTION_PATH \
    -o $OUTPUT_PATH \
    --video.index 0 \
    --batch_size 400 \
    --tracking.max_tracking 1 \
    --tracking.tracker simplemaxtracks \
    --tracking.max_tracks 17 \
    --tracking.similarity centroid \
    --tracking.match hungarian \
    --tracking.track_window 8    \
    --tracking.robust 0.95 \
    --tracking.oks_score_weighting 0 \
    --tracking.post_connect_single_breaks 1 \
    --verbosity rich \
    --no-empty-frames


# sleap-track 
# /home/mingxiao/Desktop/jellyfish/video/video_1_clips/predictions/c1_predictions.slp 
# --video.index 0 
# --video.input_format channels_last 
# --frames 0,-886 
# --batch_size 4 
# --tracking.tracker flow 
# --tracking.similarity centroid 
# --tracking.match hungarian 
# --tracking.track_window 5 
# --tracking.oks_errors  
# --tracking.oks_score_weighting 0 
# --tracking.post_connect_single_breaks 0 
# --controller_port 9000 --publish_port 9001 
# -o /home/mingxiao/Desktop/jellyfish/video/video_1_clips/predictions/predictions/c1_predictions.slp.250120_120932.predictions.slp 
# --verbosity json --no-empty-frames