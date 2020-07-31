#! /usr/bin/env bash

export CONTAINER_VOLUME="/root/ray_results"
export RUN_DB_PATH="$HOME/runs/td3"
export RUN_IMAGE_BUILD_PATH="$HOME/TD3-JAX/"
export RUN_DOCKERFILE_PATH="$RUN_IMAGE_BUILD_PATH/Dockerfile"
export RUN_IMAGE='td3'
export RUN_CONFIG_SCRIPT='config_script.py'
export RUN_CONFIG_SCRIPT_INTERPRETER='python3'
export RUN_CONFIG_SCRIPT_INTERPRETER_ARGS='-c'
export DOCKER_RUN_COMMAND='docker run -d --rm --gpus all -it'
