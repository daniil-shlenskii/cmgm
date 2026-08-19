#!/bin/bash
export HYDRA_FULL_ERROR=1
source /home/ivanov/vllm_env/bin/activate
export CUDA_VISIBLE_DEVICES=3
cd /home/ivanov/posterior-refinement

CKPT=/home/ivanov/posterior-refinement/checkpoints/sudoku_mdlm_hard/checkpoints/last.ckpt
OUTLOG=/home/ivanov/posterior-refinement/mdlm_confidence_sweep_hard.log
> "$OUTLOG"

for NFE in 1 2 3 4 6 8 16 32 64 128; do
  echo "=== NFE=$NFE ===" | tee -a "$OUTLOG"
  python -u -m main \
    hydra.run.dir=outputs/sflm_sudoku_hard/mdlm_confidence_eval_nfe${NFE} \
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
    algo=mdlm \
    strategy.find_unused_parameters=false \
    trainer.devices=1 \
    trainer.precision=bf16 \
    eval.checkpoint_path=$CKPT \
    eval.disable_ema=False \
    eval.generate_samples=True \
    eval.compute_generative_perplexity=False \
    +eval.sflm_sudoku_num_eval=2000 \
    sampling.predictor=confidence \
    sampling.steps=$NFE \
    +wandb.offline=true \
    2>&1 | tee -a "$OUTLOG" | grep -E 'acc=|Error|error'
done
