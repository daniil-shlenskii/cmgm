#!/bin/bash
source /home/ivanov/vllm_env/bin/activate
export CUDA_VISIBLE_DEVICES=2
export HYDRA_FULL_ERROR=1
cd /home/ivanov/posterior-refinement

python -u -m main \
  mode=train \
  loader.global_batch_size=512 \
  loader.eval_batch_size=256 \
  data=sudoku \
  data.cache_dir=/home/ivanov/posterior-refinement/data/sudoku_cache \
  data.difficulty=hard \
  model=mini \
  model.length=180 \
  algo=mgm \
  algo.warmup_steps=0 \
  algo.init_from_ckpt=/home/ivanov/posterior-refinement/checkpoints/sudoku_hard.ckpt \
  strategy.find_unused_parameters=true \
  trainer.devices=1 \
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
  wandb.name=mgm_fmlminit \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=20000 \
  callbacks.checkpoint_every_n_steps.save_last=True \
  checkpointing.save_dir=/home/ivanov/posterior-refinement/checkpoints/sudoku_mgm_hard_fmlminit \
  2>&1 | tee /home/ivanov/posterior-refinement/mgm_train_hard_fmlminit.log
