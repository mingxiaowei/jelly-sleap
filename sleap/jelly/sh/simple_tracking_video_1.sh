#bin/bash

JELLY_PATH="/home/mingxiao/Desktop/jellyfish/"
PREDICTION_FOLDER="${JELLY_PATH}label/multifish/predictions/"
OUTPUT_FOLDER="${JELLY_PATH}label/multifish/predictions/simple-tracking/"
OUTPUT_FILE="video_1_simple_tracking.slp"
OUTPUT_PATH="${OUTPUT_FOLDER}${OUTPUT_FILE}"

PREDICTION_FILE="multifish_animal_1_v4.slp.250117_155941.predictions.slp"
PREDICTION_PATH="${PREDICTION_FOLDER}${PREDICTION_FILE}"

if [ ! -d "$OUTPUT_FOLDER" ]; then
    mkdir -p "$OUTPUT_FOLDER"
fi

sleap-track $PREDICTION_PATH \
    -o $OUTPUT_PATH \
    --video.index 0 \
    --video.input_format channels_last \
    --batch_size 4 \
    --tracking.tracker simplemaxtracks \
    --tracking.max_tracks 17 \
    --tracking.similarity centroid \
    --tracking.match hungarian \
    --tracking.track_window 5 \
    --tracking.post_connect_single_breaks 1 \
    --verbosity json \
    --no-empty-frames

# sleap-track /home/mingxiao/Desktop/jellyfish/video/video_1_clips/predictions/c1_predictions.slp 
# --video.index 0 
# --video.input_format channels_last 
# --frames 0,-886 
# --batch_size 4 
# --tracking.tracker simple 
# --tracking.similarity centroid 
# --tracking.match hungarian 
# --tracking.track_window 5 
# --tracking.oks_errors  
# --tracking.oks_score_weighting 0 
# --tracking.post_connect_single_breaks 0 
# --controller_port 9000 --publish_port 9001 
# -o /home/mingxiao/Desktop/jellyfish/video/video_1_clips/predictions/predictions/c1_predictions.slp.250120_120039.predictions.slp 
# --verbosity json --no-empty-frames