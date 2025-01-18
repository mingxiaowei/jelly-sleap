#bin/bash
cd /home/mingxiao/Desktop/
MODEL_FOLDER="/home/mingxiao/jellyfish/label/models/250117_103127.single_instance.n=1653"
OUTPUT_PATH="original_output_a2.slp"
# VIDEO_PATH="/home/mingxiao/jellyfish/video/full_video_1.avi"
VIDEO_PATH="/home/mingxiao/jellyfish/label/animal_2_labels.v001.slp"

sleap-track -m $MODEL_FOLDER -o $OUTPUT_PATH $VIDEO_PATH --open-in-gui  
# --only-labeled-frames
# --only-suggested-frames