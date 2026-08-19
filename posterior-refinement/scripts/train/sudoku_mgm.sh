#!/bin/bash

DATA_DIR="/home/ivanov/posterior-refinement/data/sudoku_cache"  # dataset cache directory
SAVE_DIR="/home/ivanov/posterior-refinement/checkpoints/sudoku_mgm_hard"  # checkpoint output directory
DIFFICULTY="hard"                      # one of: easy, medium, hard
NUM_GPUS="${NUM_GPUS:-1}"              # set via env, e.g. NUM_GPUS=3 CUDA_VISIBLE_DEVICES=2,6,7

python -u -m main \
  mode=train \
  loader.global_batch_size=512 \
  loader.eval_batch_size=256 \
  data=sudoku \
  data.cache_dir=${DATA_DIR} \
  data.difficulty=${DIFFICULTY} \
  model=mini \
  model.length=180 \
  algo=mgm \
  strategy.find_unused_parameters=true \
  trainer.devices=${NUM_GPUS} \
  trainer.gradient_clip_val=null \
  trainer.max_steps=20000 \
  trainer.precision=bf16 \
  trainer.val_check_interval=5000 \
  trainer.limit_val_batches=1 \
  trainer.num_sanity_val_steps=1 \
  +trainer.check_val_every_n_epoch=null \
  eval.compute_generative_perplexity=False \
  eval.generate_samples=True \
  +eval.sflm_sudoku_num_eval=256 \
  sampling.steps=1 \
  +wandb.offline=true \
  wandb.project=sudoku \
  wandb.name=mgm \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=20000 \
  callbacks.checkpoint_every_n_steps.save_last=True \
  checkpointing.save_dir=${SAVE_DIR}
