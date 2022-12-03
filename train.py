"""
* The purpose of this module is to train a new network on a data set
* Basic input: a data directory
    * Options:
        * python train.py data_directory
        * python train.py data_dir --save_dir save_directory
        * python train.py data_dir --arch "vgg13"
        * python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
        * python train.py data_dir --gpu
* Output: A trained network in form of a saved checkpoint file
* Prints out:
    * training loss,
    * validation loss and
    * validation accuracy as the network trains
"""
import torch
from utils import parse_train_arguments
from utils import prepare_train_validation_data
from model import load_pretrained_model, Classifier

# 1. Read command line arguments and load data
input_args = parse_train_arguments()
data_dir = input_args.data_dir
train_loader, valid_loader = prepare_train_validation_data(data_dir, 64)

# 2. Check for gpu training based on demand and availability
if input_args.gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("-> gpu is not available on this machine, switched to cpu!")
        device = torch.device("cpu")

# 3. Load any pretrained model
arch = input_args.arch
model = load_pretrained_model(arch)


# 4. Modify pretrained model to fit our needs
if input_args.arch:
    fc_models = ["resnet", "inception"]  # Fully Connected layer
    classifier_models = ["alexnet", "vgg", "squeezenet", "densenet"]

    user_defined = Classifier(25088, 102, [6272, 1020], 0.5)
    # Find a way to get the input size for a model
    # replace purser to get list for hidden layers

if __name__ == "__main__":
    print(input_args)
