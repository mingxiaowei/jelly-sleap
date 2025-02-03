#! /bin/bash

n_threads=50
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

/home/mingxiao/micromamba/envs/tf/bin/python /home/mingxiao/Desktop/jelly-sleap/sleap/jelly/autoencoder/train_10min_cpu.py