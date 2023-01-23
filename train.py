#!/usr/bin/env python3


def build_train_pipeline(args):
    import lib as cl
    import lib.models as models

    pipe = head = cl.dataset(args.dsroot, split="train", batch_size=args.batch_size)
    mean, std = head.data_norm()
    
    model = models.create_model(args.model, head.num_classes(), force_cpu=args.force_cpu, pretrained=True, freeze=args.freeze)

    if args.mx > 0.0:
        pipe = cl.mixup(pipe, num_classes=head.num_classes(), ratio=0.5, p=args.mx)
    
    pipe = cl.augmenter(pipe, mode="train", mean=mean, std=std, size=args.image_size)
    pipe = cl.dataloader(pipe, device=model.device, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    
    pipe = cl.trainer(pipe, model, label_smoothing=args.ls, 
                lr=args.lr, weight_decay=args.wd,
                grad_clip_value=args.gcv, 
                grad_max_norm=args.gmn
                )

    batch_keys = []
    if args.grad_stats:
        pipe = cl.gradstats(pipe, model)
        batch_keys = ["gradient_min", "gradient_max", "total_norm"]
    
    pipe = cl.assessor(pipe)
    pipe = cl.averager(pipe, keymap={"correct": "accuracy", "loss": "loss"})
    
    log_writer = cl.log_writer(args.log_dir)
    pipe = cl.logger(pipe, writer=log_writer, prefix="Train", keys=["loss", "accuracy"], batch_keys=batch_keys)
    
    return pipe, head, model, log_writer


def build_vdate_pipeline(args, model, log_writer):
    import lib as cl

    pipe = head = cl.dataset(args.dsroot, split="val", batch_size=args.batch_size)
    mean, std = head.data_norm()

    pipe = cl.augmenter(pipe, mode="val", mean=mean, std=std, size=args.image_size)
    pipe = cl.dataloader(pipe, device=model.device, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    pipe = cl.validator(pipe, model)
    pipe = cl.assessor(pipe)
    pipe = cl.averager(pipe, keymap={"correct": "accuracy", "loss": "loss"})
    pipe = cl.logger(pipe, writer=log_writer, prefix="Vdate", keys=["loss", "accuracy"])
    
    return pipe, head


def run(args):
    import lib as cl
    import lib.models as models
    import torch

    tpipe, thead, model, log_writer = build_train_pipeline(args)
    vpipe, vhead = build_vdate_pipeline(args, model, log_writer)
    
    nepochs = args.num_epochs
    
    for epoch in range(nepochs):
        print(f"Epoch {epoch:02d}")
        
        # the training cycle
        thead.shuffle()
        for idx, item in cl.progress(tpipe, header="Train", end=""):
            pass
        if loss := item.get('loss', None):
            print(f" loss={loss:0.2e}", end="")
        if acc := item.get('accuracy', None):
            print(f" acc={acc:0.4f}", end="")
        print("")
        
        # the validation cycle
        with torch.no_grad():
            for idx, item in cl.progress(vpipe, header="Vdate", end=""):
                pass
        if loss := item.get('loss', None):
            print(f" loss={loss:0.2e}", end="")
        if acc := item.get('accuracy', None):
            print(f" acc={acc:0.4f}", end="")
        print("")
    
    models.save_model(model, state_dir=args.log_dir, state_name=args.run_id, epoch=epoch)


def main():
    import os, sys
    import argparse
    from datetime import datetime
    import random
    import torch
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--force-cpu', help='use the CPU even if there is a GPU', action='store_true')
    parser.add_argument('-w', '--num-workers', help='number of workers to use', type=int, default=0)
    parser.add_argument('-e', '--num-epochs', help='number of epochs', type=int, default=1)
    parser.add_argument('-b', '--batch-size', help='batch size', type=int, default=8)
    parser.add_argument('-s', '--image-size', help='resize the images before processing', type=int, default=32)
    parser.add_argument('-z', '--freeze', help='number of layers in the model to freeze', type=int, default=0)
    parser.add_argument('--mx', help='mixup augmentation probability', type=float, default=0.0)
    parser.add_argument('--ls', help='label smoothing for loss function', type=float, default=0.1)
    parser.add_argument('--gcv', '--grad-clip-value', help='gradient value clipping', type=float, default=None)
    parser.add_argument('--gmn', '--grad-max-norm', help='gradient norm clipping', type=float, default=None)
    parser.add_argument('--lr', help='learning rate for the optimizer', type=float, default=0.0001)
    parser.add_argument('--wd', help='weight decay for the optimizer', type=float, default=0.01)
    parser.add_argument('--grad-stats', help='collect and report gradient stats', action='store_true')
    parser.add_argument('model', help='the model type to use', type=str)
    parser.add_argument('dsroot', help='path to the cifar-10 or cifar-100 dataset', type=str)
    
    args = parser.parse_args()

    # prepare the log dir
    now = datetime.now()
    args.run_id = now.strftime("%Y%m%d-%H%M%S")
    args.log_dir = os.path.join("snapshots", f"{args.run_id}-train")
    
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

