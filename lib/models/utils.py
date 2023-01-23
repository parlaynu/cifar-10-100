import os
import torch

from .mobilenet_v3 import mobilenet_v3_small, mobilenet_v3_large
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152


models = {
    "mobilenet_v3_small": mobilenet_v3_small,
    "mobilenet_v3_large": mobilenet_v3_large,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152
}

def create_model(name, num_classes, *, force_cpu, pretrained, freeze):
    return models[name](num_classes, force_cpu=force_cpu, pretrained=pretrained, freeze=freeze)


def load_model(state_file, *, use_gpu):
    print(f"loading model state from {state_file}")

    # load the state
    state = torch.load(state_file, map_location=torch.device('cpu'))
    
    # create the model instance
    epoch = state['epoch']
    name = state['name'].split('.')[-1]
    num_classes = state['num_classes']
    
    model = create_model(name, num_classes, use_gpu=use_gpu, pretrained=False)
    
    # update the model state
    model.load_state_dict(state['model'])
    
    return model, epoch


def save_model(model, *, state_dir, state_name, epoch):
    # create the state file name
    state_file = os.path.join(state_dir, f"{state_name}-{epoch:02d}.pt")
    print(f"saving model state to {state_file}")

    # make sure the directory exists
    if not os.path.exists(state_dir):
        os.makedirs(state_dir, exist_ok=True)

    # save all the state
    state = { 
        'epoch': epoch,
        'name': model.fullname,
        'num_classes': model.num_outputs,
        'model': model.state_dict()
    }
    torch.save(state, state_file)
    
    return state_file

