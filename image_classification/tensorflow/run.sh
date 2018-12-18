#/bin/bash

RANDOM_SEED=1
QUALITY=95
set -e

# Register the model as a source root
export PYTHONPATH="$(pwd):${PYTHONPATH}"

MODEL_DIR="/tmp/resnet_imagenet_${RANDOM_SEED}"

python3 official/resnet/imagenet_main.py $RANDOM_SEED --use_synthetic_data \
  --model_dir $MODEL_DIR --train_epochs 8 --stop_threshold $QUALITY --batch_size 1 \
  --version 1 --resnet_size 50 --epochs_between_evals 4 --num_gpus 0 --max_train_steps 100 --hooks "ExamplesPerSecondHook"

# To run on 8xV100s, instead run:
#python3 official/resnet/imagenet_main.py $RANDOM_SEED --data_dir /imn/imagenet/combined/ \
#   --model_dir $MODEL_DIR --train_epochs 10000 --stop_threshold $QUALITY --batch_size 1024 \
#   --version 1 --resnet_size 50 --dtype fp16 --num_gpus 8 \
#   --epochs_between_evals 4
