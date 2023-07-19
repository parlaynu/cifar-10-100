#!/usr/bin/env bash

# cifar-100 at 32x32

../../find-lr.py -w 2 -e 4 -b 64 -f 0.01 mobilenet_v3_large ~/Projects/datasets/cifar-100

../../train.py -w 2 -e 75 -b 64 --lr 0.0001 mobilenet_v3_large ~/Projects/datasets/cifar-100
../../train.py -w 2 -e 75 -b 64 --lr 0.0002 mobilenet_v3_large ~/Projects/datasets/cifar-100

# cifar-10 at 224x224

../../find-lr.py -w 2 -e 4 -b 64 -s 224 -f 0.001 mobilenet_v3_large ~/Projects/datasets/cifar-10
../../find-lr.py -w 2 -e 4 -b 64 -s 224 -f 0.0001 mobilenet_v3_large ~/Projects/datasets/cifar-10

../../train.py -w 2 -e 75 -b 64 -s 224 --lr 0.00001 mobilenet_v3_large ~/Projects/datasets/cifar-10
../../train.py -w 2 -e 75 -b 64 -s 224 --lr 0.00005 mobilenet_v3_large ~/Projects/datasets/cifar-10
../../train.py -w 2 -e 75 -b 64 -s 224 --lr 0.0001 mobilenet_v3_large ~/Projects/datasets/cifar-10

