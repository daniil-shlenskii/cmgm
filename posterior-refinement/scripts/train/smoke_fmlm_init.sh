#!/bin/bash
source /home/ivanov/vllm_env/bin/activate
export CUDA_VISIBLE_DEVICES=2
export HYDRA_FULL_ERROR=1
cd /home/ivanov/posterior-refinement

python -u -m main \
  mode=train \
  loader.global_batch_size=64 \
  loader.batch_size=64 \
  loader.eval_batch_size=32 \
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
  trainer.max_steps=15 \
  trainer.precision=bf16 \
  trainer.val_check_interval=1000000 \
  trainer.num_sanity_val_steps=0 \
  +trainer.check_val_every_n_epoch=null \
  eval.compute_generative_perplexity=False \
  eval.generate_samples=True \
  +eval.sflm_sudoku_num_eval=32 \
  sampling.steps=1 \
  +wandb.offline=true \
  wandb.project=sudoku_smoke \
  wandb.name=mgm_fmlm_init_smoke \
  callbacks.checkpoint_every_n_steps.every_n_train_steps=1000000 \
  checkpointing.save_dir=/home/ivanov/posterior-refinement/checkpoints/sudoku_mgm_fmlm_init_smoke \
  2>&1 | tee /tmp/mgm_fmlm_init_smoke.log
