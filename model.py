"""
* The purpose of this module is:
    * to load a pre-trained neural network model
    * to replace feed forward configuration, if asked
    * to replace default values for hyperparameters
"""
import torch
from sys import exit
from torch import nn
import torch.nn.functional as f


def load_pretrained_model(model_name="vgg16_bn"):

    all_models = torch.hub.list("pytorch/vision", force_reload=True)

    if model_name in all_models or model_name == "inception":
        print(f"-> Loading pretrained model: {model_name}")

        model = torch.hub.load("pytorch/vision", model_name, pretrained=True)
        # model = getattr(torchvision.models, model_name)(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        return model
    else:
        exit(
            f"-> No such model architecture!\n"
            f"-> Check your model name for typo and try again!\n"
            f"-> For models: resnet, inception, alexnet, vgg, squeezenet, densenet\n"
            f"-> inception model expects (299,299) sized images, won't work for now!"
            f"-> Find your variant: {all_models}"
        )


class Classifier(nn.Module):
    def __init__(self, layer_in, layer_out, layers_inside, drop_p=0.2):
        super().__init__()
        self.layers_inside = nn.ModuleList([nn.Linear(layer_in, layers_inside[0])])

        # Add a variable number of more hidden layers
        layer_sizes = zip(layers_inside[:-1], layers_inside[1:])
        self.layers_inside.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

        self.output = nn.Linear(layers_inside[-1], layer_out)

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        for each in self.layers_inside:
            x = f.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)

        return f.log_softmax(x, dim=1)
