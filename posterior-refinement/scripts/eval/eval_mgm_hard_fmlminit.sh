#!/bin/bash
export HYDRA_FULL_ERROR=1
source /home/ivanov/vllm_env/bin/activate
export CUDA_VISIBLE_DEVICES=0
cd /home/ivanov/posterior-refinement

python -u -m main \
  hydra.run.dir=outputs/sflm_sudoku_hard/mgm_fmlminit_eval \
  mode=ppl_eval \
  seed=1 \
  loader.global_batch_size=200 \
  loader.batch_size=200 \
  loader.eval_batch_size=200 \
  loader.num_workers=8 \
  data=sudoku \
  data.cache_dir=/home/ivanov/posterior-refinement/data/sudoku_cache \
  data.difficulty=hard \
  model=mini \
  model.length=180 \
  algo=mgm \
  strategy.find_unused_parameters=true \
  trainer.devices=1 \
  trainer.gradient_clip_val=null \
  trainer.precision=bf16 \
  eval.checkpoint_path=/home/ivanov/posterior-refinement/checkpoints/sudoku_mgm_hard_fmlminit/checkpoints/last.ckpt \
  eval.disable_ema=False \
  eval.generate_samples=True \
  eval.compute_generative_perplexity=False \
  +eval.sflm_sudoku_num_eval=2000 \
  sampling.steps=1 \
  +wandb.offline=true \
  2>&1 | tee /home/ivanov/posterior-refinement/mgm_eval_hard_fmlminit.log
