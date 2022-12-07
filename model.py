"""
* The purpose of this module is:
    * to load a pre-trained neural network model
    * to replace feed forward configuration, if asked
    * to replace default values for hyperparameters
"""
import time
import torch
from sys import exit
from torch import nn, optim
import torch.nn.functional as f
from torchvision import models


def load_pretrained_model(model_name="vgg16_bn"):
    # all_models = torch.hub.list("pytorch/vision", force_reload=True)
    # if model_name in all_models and model_name != "inception":
    # print(f"-> Loading pretrained model: {model_name}")

    # model = getattr(torchvision.models, model_name)(pretrained=True)
    # model = torch.hub.load("pytorch/vision", model_name, pretrained=True)

    a_few_models = ["vgg13", "vgg16_bn", "vgg19", "resnet34", "resnet152", "densenet201"]
    if model_name in a_few_models:
        print(f"-> Loading pretrained model: {model_name}")

    if model_name == 'vgg13':
        model = models.vgg13(pretrained=True)

    elif model_name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)

    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)

    # small model to run parse options
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)

    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)

    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=True)

    else:
        exit(
            f"-> No such model architecture!\n"
            f"-> Check your model name for typo and try again!\n"
            f"-> Please select from: {a_few_models}"
        )

    # Freeze feature parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


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


def train_model(e, model, criterion, optimizer, t_loader, v_loader, device):
    epochs = e
    batches = 0
    print_every = 100
    train_loss = 0
    train_losses, valid_losses = [], []

    print(f"-> Train the model started... ")
    start = time.time()
    for e in range(epochs):
        for images, labels in t_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Feedforward
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)

            # Back propagation
            optimizer.zero_grad()

            # loss.requires_grad = True
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            batches += 1  # Training batches
            # print(batches, end=" ")

            # Test performance in every few batches
            if batches % print_every == 0:
                model.eval()
                valid_loss, accuracy = 0, 0

                with torch.no_grad():
                    for images_, labels_ in v_loader:
                        images_, labels_ = images_.to(device), labels_.to(device)

                        log_ps = model(images_)
                        loss = criterion(log_ps, labels_)

                        valid_loss += loss.item()

                        # Calculation of accuracy
                        ps = torch.exp(log_ps)
                        top_ps, top_class = ps.topk(1, dim=1)
                        equality = top_class == labels_.view(*top_class.shape)
                        accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

                    t_loss = round(train_loss / print_every, 2)
                    v_loss = round(valid_loss / len(v_loader), 2)

                    train_losses.append(t_loss)
                    valid_losses.append(v_loss)

                    print(
                        f"\t$ {e + 1}/{epochs}.. Batch:{batches}.. "
                        f"Train_loss:{t_loss}.. "
                        f"Valid_loss:{v_loss}.. "
                        f"Accuracy: {round(accuracy / len(v_loader), 2)}"
                    )

                    model.train()
                    train_loss = 0

    print(f"\n-> Train ended in : {round((time.time() - start) / 60, 1)} minutes.")

    return train_losses, valid_losses


def check_accuracy(model, testset, device):
    start = time.time()
    model.eval()
    batch = 0

    accuracy = 0
    with torch.no_grad():
        for images, labels in testset:
            lpt = time.time()

            images, labels = images.to(device), labels.to(device)
            log_ps = model.forward(images)

            # Calculation of accuracy
            ps = torch.exp(log_ps)
            top_ps, top_class = ps.topk(1, dim=1)
            equality = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
            batch += 1

            print(f"\tBatch #{batch}.. {round(time.time() - lpt, 1)} seconds.")

        print(f"\n-> Accuracy on the test dataset: {round(accuracy / len(testset), 2)}")
        print(f"-> Test done in: {round(time.time() - start, 2)} seconds.")


def rebuild_the_model(pth_path, gpu):
    # Loading checkpoint
    if torch.cuda.is_available():
        ml = "cuda:0"
    else:
        ml = 'cpu'
    checkpoint = torch.load(pth_path, map_location=ml)

    # Load the pre-trained model
    model = load_pretrained_model(checkpoint["arch"])
    for param in model.parameters():
        param.requires_grad = False

    # Substitute the classifier and other details
    first_layer = checkpoint["input_size"]
    output_size = checkpoint["output_size"]
    layers_inside = checkpoint["layers_inside"]
    dp = checkpoint["dropout_p"]

    if checkpoint["model_type"] == "classifier":
        model.classifier = Classifier(first_layer, output_size, layers_inside, dp)
    if checkpoint["model_type"] == "fc":
        model.fc = Classifier(first_layer, output_size, layers_inside, dp)

    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_index"]

    epochs = checkpoint['epochs']
    learning_rate = checkpoint['learning_rate']

    # Return saved values for other parts
    criterion = checkpoint['criterion_state_dict']
    optimizer = checkpoint['optimizer_state_dict']

    if gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model.to(device)
            print(f"-> Activating gpu done.")
        else:
            print(f"-> pgu is not available!")

    print(f"-> Rebuilding the model {checkpoint['arch']} done!")
    return model, epochs, learning_rate, optimizer, criterion


def rebuild_simple(pth_path, gpu):
    if torch.cuda.is_available():
        ml = "cuda:0"
    else:
        ml = 'cpu'
    checkpoint = torch.load(pth_path, map_location=ml)

    # Load the pre-trained model
    model = load_pretrained_model(checkpoint["arch"])
    for param in model.parameters():
        param.requires_grad = False

    # Unpacking checkpoint
    first_layer = checkpoint["input_size"]
    output_size = checkpoint["output_size"]
    layers_inside = checkpoint["layers_inside"]
    dp = checkpoint["dropout_p"]

    if checkpoint["model_type"] == "classifier":
        model.classifier = Classifier(first_layer, output_size, layers_inside, dp)
    if checkpoint["model_type"] == "fc":
        model.fc = Classifier(first_layer, output_size, layers_inside, dp)

    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_index"]

    if gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model.to(device)
            print(f"-> GPU Activated.")
        else:
            print(f"-> GPU is not available!")

    print(f"-> Rebuilding the model {checkpoint['arch']} done!")
    return model
