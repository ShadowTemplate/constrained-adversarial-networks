#!/usr/bin/env bash
# run from src
experiment_name=$1
echo $experiment_name
bgans_path="../out/images/$experiment_name.zip"
echo $bgans_path
checkpoints_path="../out/model_checkpoints/$experiment_name.zip"
echo $checkpoints_path
tensorboard_path="../out/tensorboard/$experiment_name.zip"
echo $tensorboard_path
zip $experiment_name.zip $bgans_path $checkpoints_path $tensorboard_path ./can.log
