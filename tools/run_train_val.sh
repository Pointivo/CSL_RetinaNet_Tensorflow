#!/bin/bash -v

for i in {1..5}
do
  python multi_gpu_train.py
  python run_validation.py --data_dir /home/faisal/python-microservices/image-recognition/image_recognition/tmp/val_30 --gpu 0
done