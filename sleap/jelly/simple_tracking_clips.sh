#bin/bash
cd /home/mingxiao/Desktop/jellyfish/
PREDICTION_FOLDER="video/video_1_clips/predictions/"
OUTPUT_FOLDER="video/video_1_clips/predictions/simple-tracking"

if [ ! -d "$OUTPUT_FOLDER" ]; then
    mkdir -p "$OUTPUT_FOLDER"
fi

for i in $(seq 1 6)
do
    PREDICTION_PATH="${PREDICTION_FOLDER}c${i}_predictions.slp"
    OUTPUT_PATH="${OUTPUT_FOLDER}c${i}_simple_tracking.slp"
    sleap-track $PREDICTION_PATH \
        -o $OUTPUT_PATH \
        --video.index 0 \
        --video.input_format channels_last \
        --frames 0,-886 \
        --batch_size 4 \
        --tracking.tracker simple \
        --tracking.similarity centroid \
        --tracking.match hungarian \
        --tracking.track_window 5 \
        # --tracking.oks_errors \
        --tracking.oks_score_weighting 0 \
        --tracking.post_connect_single_breaks 0 \
        --verbosity json \
        --no-empty-frames
done

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