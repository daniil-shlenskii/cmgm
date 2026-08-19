#!/bin/bash
export HYDRA_FULL_ERROR=1

DATA_DIR="/home/ivanov/posterior-refinement/data/sudoku_cache"                  # dataset cache directory
CKPT="/home/ivanov/posterior-refinement/checkpoints/sudoku_mgm_hard/checkpoints/last.ckpt"  # trained MGM checkpoint
DIFFICULTY="hard"                # must match DIFFICULTY used in sudoku_mgm.sh training

python -u -m main \
  hydra.run.dir=outputs/sflm_sudoku_hard/mgm_sflm_sudoku_hard_eval \
  mode=ppl_eval \
  seed=1 \
  loader.global_batch_size=200 \
  loader.batch_size=200 \
  loader.eval_batch_size=200 \
  loader.num_workers=8 \
  data=sudoku \
  data.cache_dir=${DATA_DIR} \
  data.difficulty=${DIFFICULTY} \
  model=mini \
  model.length=180 \
  algo=mgm \
  strategy.find_unused_parameters=true \
  trainer.devices=1 \
  trainer.gradient_clip_val=null \
  trainer.precision=bf16 \
  eval.checkpoint_path=${CKPT} \
  eval.disable_ema=False \
  eval.generate_samples=True \
  eval.compute_generative_perplexity=False \
  +eval.sflm_sudoku_num_eval=2000 \
  sampling.steps=1 \
  +wandb.offline=true
