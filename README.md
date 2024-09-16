# 317VO

## here is huanyao's second project...

## setup

### dataset preparing

structure:
cityscapes
    - train
        - raw
            - zurich_000121_000019_leftImg8bit.png
            - ...
        - seg
            - zurich_000121_000019_gtFine_color.png
            - ...
        - sp
            - zurich_000121_000019_spgt.np
            - ...
    - val
        - ...
    - test
        - ...

### environment
python 3.9 CUDA 12.2 Nvidia GTR 4090

conda create -n py39-317 python=3.9
conda activate py39-317

install pytorch from https://pytorch.org/

torch 2.4.0
opencv 4.10.0.84

## train networks on your dataset

## pretrianed

## run demo

## citations