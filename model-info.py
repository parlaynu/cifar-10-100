#!/usr/bin/env python3
import argparse
import lib.models as models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='the model type to use', type=str)
    
    args = parser.parse_args()
    
    model = models.create_model(args.model, 10, use_gpu=False, pretrained=False, freeze=0)
    print(model)


if __name__ == "__main__":
    main()

