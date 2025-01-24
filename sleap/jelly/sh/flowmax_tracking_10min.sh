#bin/bash
cd /home/mingxiao/Desktop/jellyfish/
PREDICTION_FOLDER="video/video_1_clips/predictions/"
OUTPUT_FOLDER="video/video_1_clips/predictions/flowmax-tracking/"

if [ ! -d "$OUTPUT_FOLDER" ]; then
    mkdir -p "$OUTPUT_FOLDER"
fi

n_threads=40
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

PREDICTION_PATH="/home/mingxiao/Desktop/jellyfish/video/video_1_clips/predictions/flowmax-tracking/reencoded_tracking_10min_flowmax.slp"
OUTPUT_PATH="${OUTPUT_FOLDER}reencoded_10min.slp"
sleap-track $PREDICTION_PATH \
    -o $OUTPUT_PATH \
    --video.index 0 \
    --video.input_format channels_last \
    --batch_size 4 \
    --tracking.max_tracking 1 \
    --tracking.tracker flowmaxtracks \
    --tracking.max_tracks 17 \
    --tracking.similarity centroid \
    --tracking.match hungarian \
    --tracking.track_window 5 \
    --tracking.oks_score_weighting 0 \
    --tracking.post_connect_single_breaks 0 \
    --verbosity json \
    --no-empty-frames