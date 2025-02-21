#! /bin/bash

n_threads=40
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

sleap-track \
    /home/mingxiao/Desktop/jellyfish/label/multifish/multifish_animal_1_v10.slp \
    --video.index 0 \
    --batch_size 400 \
    --frames 0,-3239999 \
    -m /home/mingxiao/Desktop/jellyfish/label/multifish/models/250221_022817.centroid.n=4149 \
    -m /home/mingxiao/Desktop/jellyfish/label/multifish/models/250221_031000.centered_instance.n=4149 \
    --controller_port 40271 \
    --publish_port 40951 \
    -o /home/mingxiao/Desktop/jellyfish/label/multifish/predictions/multifish_animal_1_v10.slp.250221_051111.predictions.slp \
    --verbosity json \
    --no-empty-frames