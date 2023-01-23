#!/usr/bin/env python3


def build_pipeline(args):
    import lib as cl
    import lib.models as models

    pipe = head = cl.dataset(args.dsroot, split="val", batch_size=args.batch_size)
    mean, std = head.data_norm()

    model, epoch = models.load_model(args.state_file, force_cpu=args.force_cpu, pretrained=False, freeze=0)

    pipe = cl.augmenter(pipe, mode="val", mean=mean, std=std, size=args.image_size)
    pipe = cl.dataloader(pipe, device=model.device, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    
    pipe = cl.validator(pipe, model)
    pipe = cl.assessor(pipe)
    
    return pipe, head


def run(args):
    import os
    import torch
    import lib as cl
    import lib.models as models

    pipe, head = build_pipeline(args)
    
    report_file = os.path.join(args.log_dir, "report.csv")
    report = open(report_file, "w")
    print("file name,class,prediction,correct,confidence", file=report)
    
    total_preds = 0
    total_correct = 0
    
    with torch.no_grad():
        for idx, item in cl.progress(pipe, header="Report", end=""):
            names = item['name']
            labels = item['label'].tolist()
            preds = item['pred-label'].tolist()
            probs = item['pred-prob'].tolist()
            
            for name, label, pred, prob in zip(names, labels, preds, probs):
                label_name = head.class_name(label)
                pred_name = head.class_name(pred)
                
                total_preds += 1
                
                correct = 1 if label == pred else 0
                total_correct += correct
                    
                print(f"{name},{label_name},{pred_name},{correct},{prob:0.4f}", file=report)
    
    if loss := item.get('loss', None):
        print(f" loss={loss:0.2e}", end="")
    if acc := item.get('accuracy', None):
        print(f" acc={acc:0.4f}", end="")
    print("")

    report.close()
    
    correct_percentage = total_correct / total_preds * 100.0
    print(f"total predictions: {total_preds}")
    print(f"    total correct: {total_correct} {correct_percentage:0.2f}%")


def main():
    import os, sys
    import argparse
    from datetime import datetime
    import random
    import torch
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--force-cpu', help='use the CPU even if there is a GPU', action='store_true')
    parser.add_argument('-w', '--num-workers', help='number of workers to use', type=int, default=0)
    parser.add_argument('-b', '--batch-size', help='batch size', type=int, default=8)
    parser.add_argument('-s', '--image-size', help='resize the images before processing', type=int, default=32)
    parser.add_argument('state_file', help='the model state to load', type=str)
    parser.add_argument('dsroot', help='path to the cifar-10 or cifar-100 dataset', type=str)
    
    args = parser.parse_args()
    
    # prepare the log dir
    now = datetime.now()
    args.run_id = now.strftime("%Y%m%d-%H%M%S")
    args.log_dir = os.path.join("snapshots", f"{args.run_id}-report")
    
    os.makedirs(args.log_dir)
    
    # save the command line
    cmd_file = os.path.join(args.log_dir, "command.txt")
    with open(cmd_file, "w") as f:
        print(" ".join(sys.argv), file=f)
    
    # initialize the random number generators
    torch.manual_seed(1330)
    random.seed(1330)

    # and run...
    run(args)    


if __name__ == "__main__":
    main()

