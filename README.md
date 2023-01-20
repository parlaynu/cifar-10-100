# CIFAR 10/100 Experiments

A simple setup for experimenting with the CIFAR 10/100 datasets.

* https://www.cs.toronto.edu/~kriz/cifar.html

## Tools

The tools are based around building a pipeline of generators chained together with each node performing
a well defined function. To see it in action, see the functions `build_train_pipeline` and `build_vdate_pipeline`
in the file `train.py`. It's all pretty easy to follow and understand... at least I think it is.

### Training

The tool for training is `train.py`.

    $ ./train.py -h
    usage: train.py [-h] [-c] [-w NUM_WORKERS] [-e NUM_EPOCHS] [-b BATCH_SIZE] [--mx MX] [--ls LS] [--gcv GCV]
                    [--gmn GMN] [--lr LR] [--wd WD] [--grad-stats]
                    model dsroot
                    
    positional arguments:
      model                 the model type to use
      dsroot                path to the cifar-10 or cifar-100 dataset
      
    options:
      -h, --help            show this help message and exit
      -c, --use-cpu         use the CPU even if there is a GPU
      -w NUM_WORKERS, --num-workers NUM_WORKERS
                            number of workers to use
      -e NUM_EPOCHS, --num-epochs NUM_EPOCHS
                            number of epochs
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            batch size
      --mx MX               mixup augmentation probability
      --ls LS               label smoothing for loss function
      --gcv GCV, --grad-clip-value GCV
                            gradient value clipping
      --gmn GMN, --grad-max-norm GMN
                            gradient norm clipping
      --lr LR               learning rate optimizer
      --wd WD               weight decay for optimizer
      --grad-stats          collect and report gradient stats

The model types supported can be seen in the file `lib/models/utils.py`.

The training output is written to tensorboard files in a subdirectory of `snapshots`. The files written 
there are:

* the full command line used to invoke train.py - command.txt
* the tensorboard log file
* a checkpoint of the model weights at the end of training

The checkpoint file contains enough information to run the report generator. 

### Find Learning Rate

The tool `find-lr.py` implements an algorithm based on the "LR Range Test" from the paper
[Cyclical Learning Rates for Training Neural Networks](https://arxiv.org/abs/1506.01186) written by Leslie N. Smith 


    $ ./find-lr.py -h
    usage: find-lr.py [-h] [-c] [-w NUM_WORKERS] [-e NUM_EPOCHS] [-b BATCH_SIZE] [-i INITIAL_LR] [-f FINAL_LR]
                      model dsroot
                      
    positional arguments:
      model                 the model type to use
      dsroot                path to the cifar-10 or cifar-100 dataset
      
    options:
      -h, --help            show this help message and exit
      -c, --use-cpu         use the CPU even if there is a GPU
      -w NUM_WORKERS, --num-workers NUM_WORKERS
                            number of workers to use
      -e NUM_EPOCHS, --num-epochs NUM_EPOCHS
                            number of epochs
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            batch size
      -i INITIAL_LR, --initial-lr INITIAL_LR
                            initial learning rate
      -f FINAL_LR, --final-lr FINAL_LR
                            final learning rate


### Reporting

The tool `report.py` proceses the training checkpoing data and generates a csv with the details results
of using the model on the test data.

    $ ./report.py  -h
    usage: report.py [-h] [-c] [-w NUM_WORKERS] [-b BATCH_SIZE] state_file dsroot
    
    positional arguments:
      state_file            the model state to load
      dsroot                path to the cifar-10 or cifar-100 dataset
      
    options:
      -h, --help            show this help message and exit
      -c, --use-cpu         use the CPU even if there is a GPU
      -w NUM_WORKERS, --num-workers NUM_WORKERS
                            number of workers to use
      -b BATCH_SIZE, --batch-size BATCH_SIZE
                            batch size

### Data Statistics

The script `dset-info.py` was used to get the mean and std of the datasets which is used for normalizing the
data in the data training pipeline.

    $ ./dset-info.py  -h
    usage: dset-info.py [-h] dsroot
    
    positional arguments:
      dsroot      path to the cifar-10 or cifar-100 dataset
      
    options:
      -h, --help  show this help message and exit

The results from this have been incorporated into the dataset classes for each dataset and are automatically
used by the trainer.

