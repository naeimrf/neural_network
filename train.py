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
import os

import torch
from torch import nn, optim
from utils import parse_train_arguments
from utils import prepare_train_validation_data, prepare_test_data
from model import load_pretrained_model, Classifier, train_model, check_accuracy

# 1. Read command line arguments and load data
input_args = parse_train_arguments()
data_dir = input_args.data_dir
train_loader, valid_loader, train_data, _ = prepare_train_validation_data(data_dir, 64)

# 2. Check for gpu training based on demand and availability
device = torch.device("cpu")
if input_args.gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"-> GPU activated. Number of available units: {torch.cuda.device_count()}")
    else:
        print("-> gpu was not available, switched to cpu!")

# 3. Load any pretrained model
arch = input_args.arch
model = load_pretrained_model(arch)

# 4. Modify pretrained model to fit our needs
if not bool(input_args.hidden_units):
    input_args.hidden_units.append(510)

model_type = None
first_layer = 0
fc_models = ["resnet", "inception"]  # Fully Connected layer
for fcm in fc_models:
    if fcm in arch:
        first_layer = model.state_dict()["fc.weight"].numpy().shape[1]
        model.fc = Classifier(first_layer, 102, input_args.hidden_units, 0.5)
        model_type = "fc"
        print(f"-> Feedforward fc is replaced with:\n{model.fc}")

classifier_models = ["alexnet", "vgg", "squeezenet", "densenet"]
for cm in classifier_models:
    if cm in arch:
        first_layer = model.state_dict()["classifier.0.weight"].numpy().shape[1]
        model.classifier = Classifier(first_layer, 102, input_args.hidden_units, 0.5)
        model_type = "classifier"
        print(f"-> Feedforward classifier is replaced with:\n{model.classifier}")

if first_layer == 0:
    exit(
        "Unit size for first layer in the selected model failed! Check train mmodule, part #4!"
    )

# 5. Train the model
print(f"-> Training with:\n\t* Epochs: {input_args.epochs}\n\t* Learning_rate: {input_args.learning_rate}")
criterion = nn.NLLLoss()
if model_type == "classifier":
    optimizer = optim.Adam(model.classifier.parameters(), lr=input_args.learning_rate)
else:
    optimizer = optim.Adam(model.fc.parameters(), lr=input_args.learning_rate)

model.to(device)
train_model(
    input_args.epochs, model, criterion, optimizer, train_loader, valid_loader, device
)

# 6. Check network accuracy on test dataset
test_loader = prepare_test_data(data_dir, 64)
check_accuracy(model, test_loader, device)

# 7. Save checkpoint for trained model
model.class_to_idx = train_data.class_to_idx
if model_type == "classifier":
    hidden_layers = [each.out_features for each in model.classifier.layers_inside]
else:
    hidden_layers = [each.out_features for each in model.fc.layers_inside]

checkpoint = {
    "arch": input_args.arch,
    "input_size": first_layer,
    "output_size": 102,
    "layers_inside": hidden_layers,
    "dropout_p": 0.5,
    "class_to_index": train_data.class_to_idx,
    "state_dict": model.state_dict(),
    "optim_dict": optimizer.state_dict(),
    "model_type": model_type,
    "crit_dict": criterion.state_dict(),
    "learning_rate": input_args.learning_rate,
}

if input_args.save_dir:
    if not os.path.exists(input_args.save_dir):
        os.makedirs(input_args.save_dir)
    torch.save(checkpoint, f"{input_args.save_dir}/checkpoint.pth")
    print(f"-> Files in target folder: {os.listdir(input_args.save_dir)}")
else:
    torch.save(checkpoint, "checkpoint.pth")
    print(f"-> Files in current path: {os.listdir(os.getcwd())}")
