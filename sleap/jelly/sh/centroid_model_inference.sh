#bin/bash
cd /home/mingxiao/Desktop/
MODEL_FOLDER="/home/mingxiao/jellyfish/label/multifish/models/250117_104940.centroid.n=1653"
OUTPUT_PATH="centroid_output_1.slp"
# VIDEO_PATH="/home/mingxiao/jellyfish/video/full_video_1.avi"
VIDEO_PATH="/home/mingxiao/jellyfish/label/multifish/multifish_animal_1_v3.slp"

sleap-track -m $MODEL_FOLDER -o $OUTPUT_PATH $VIDEO_PATH --open-in-gui  
# --only-labeled-frames
# --only-suggested-frames