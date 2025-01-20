#bin/bash
cd /home/mingxiao/Desktop/jellyfish/
CENTROID_MODEL_FOLDER="label/multifish/models/250117_154237.centroid.n=1653"
CENTERED_INSTANCE_MODEL_FOLDER="label/multifish/models/250117_155604.centered_instance.n=1653"
OUTPUT_FOLDER="video/video_1_clips/predictions/"
VIDEO_FOLDER="video/video_1_clips/"

if [ ! -d "$OUTPUT_FOLDER" ]; then
    mkdir -p "$OUTPUT_FOLDER"
fi

for i in $(seq 1 6)
do
    VIDEO_PATH="${VIDEO_FOLDER}full_video_1_c${i}.mp4"
    OUTPUT_PATH="${OUTPUT_FOLDER}c${i}_predictions.slp"
    sleap-track \
        -m $CENTROID_MODEL_FOLDER \
        -m $CENTERED_INSTANCE_MODEL_FOLDER \
        -o $OUTPUT_PATH $VIDEO_PATH 
done

# sleap-track -m $MODEL_FOLDER -o $OUTPUT_PATH $VIDEO_PATH --open-in-gui  
# --only-labeled-frames
# --only-suggested-frames