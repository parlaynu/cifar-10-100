#!/usr/bin/env bash

../../cifar-10/find-lr.py -w 2 -e 2 -b 64 -s 224 -z 0 -f 0.0005 resnet50 ~/Projects/datasets/cifar-10
../../cifar-10/find-lr.py -w 2 -e 2 -b 64 -s 224 -z 4 -f 0.0005 resnet50 ~/Projects/datasets/cifar-10

../../cifar-10/train.py -w 2 -e 50 -b 64 --lr 0.000005 -s 224 -z 0 resnet50 ~/Projects/datasets/cifar-10

../../cifar-10/train.py -w 2 -e 50 -b 64 --lr 0.00001 -s 224 -z 0 resnet50 ~/Projects/datasets/cifar-10
../../cifar-10/train.py -w 2 -e 50 -b 64 --lr 0.00001 -s 224 -z 4 resnet50 ~/Projects/datasets/cifar-10
../../cifar-10/train.py -w 2 -e 50 -b 64 --lr 0.00001 -s 224 -z 5 resnet50 ~/Projects/datasets/cifar-10
../../cifar-10/train.py -w 2 -e 50 -b 64 --lr 0.00001 -s 224 -z 6 resnet50 ~/Projects/datasets/cifar-10

