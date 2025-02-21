import torch
from options.opts import get_training_arguments
from cvnets import get_model

def save_model(*args, **kwargs):
    opts = get_training_arguments()
    model = get_model(opts)
    torch.save(model, 'model_structure_ori.pt')

if __name__ == "__main__":
    save_model()