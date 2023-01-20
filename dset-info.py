#!/usr/bin/env python3
import numpy as np

import lib as cl


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dsroot', help='path to the cifar-10 or cifar-100 dataset', type=str)
    
    args = parser.parse_args()
    
    # create the dataset object
    dset = cl.dataset(args.dsroot, split="train", batch_size=1)
    
    print(f"dataset length: {len(dset)}")
    
    for idx, item in enumerate(dset):
        image = item['image']
        
        if idx == 0:
            m = np.mean(image, axis=(0,1))
            s = np.std(image, axis=(0,1))
            continue
        
        m += np.mean(image, axis=(0,1))
        s += np.std(image, axis=(0,1))
        
    m = m / (idx+1) / 255.0
    s = s / (idx+1) / 255.0
    
    print(f"mean: {m}")
    print(f" std: {s}")

if __name__ == "__main__":
    main()

