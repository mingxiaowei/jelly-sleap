#! /bin/bash

n_threads=40
export OMP_NUM_THREADS=$n_threads
export OPENBLAS_NUM_THREADS=$n_threads
export MKL_NUM_THREADS=$n_threads
export NUMEXPR_NUM_THREADS=$n_threads
export VECLIB_MAXIMUM_THREADS=$n_threads

/home/mingxiao/micromamba/envs/sleap/bin/python python/single_model_eval.py