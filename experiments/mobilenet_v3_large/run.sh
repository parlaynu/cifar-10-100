#!/usr/bin/env bash

../../find-lr.py -w 2 -e 4 -b 64 -f 0.01 mobilenet_v3_large ~/Projects/datasets/cifar-100/

../../train.py -w 2 -e 75 -b 64 --lr 0.0001 mobilenet_v3_large ~/Projects/datasets/cifar-100
../../train.py -w 2 -e 75 -b 64 --lr 0.0002 mobilenet_v3_large ~/Projects/datasets/cifar-100

