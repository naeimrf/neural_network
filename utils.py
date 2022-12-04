"""
* The purpose of this module is:
    * to read command line arguments,
    * to load data and
    * to preprocess the images
"""
import os
import json
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


def parse_train_arguments():
    parser_train = argparse.ArgumentParser(
        description="*** Parse the training arguments ***",
    )

    print(parser_train.description)
    parser_train.add_argument("data_dir", action="store")

    # use -- to define optional arguments
    parser_train.add_argument(
        "--save_dir", action="store", dest="save_dir", default=os.getcwd()
    )
    parser_train.add_argument("--arch", action="store", dest="arch", default="vgg16_bn")
    parser_train.add_argument(
        "--learning_rate",
        action="store",
        dest="learning_rate",
        default=0.001,
        type=float,
    )
    parser_train.add_argument(
        "--hidden_units",
        action="append",
        dest="hidden_units",
        type=int,
        default=[],
    )
    parser_train.add_argument("--gpu", action="store_true", default=False)
    parser_train.add_argument(
        "--epochs", action="store", dest="epochs", type=int, default=10
    )

    return parser_train.parse_args()


def parse_predict_arguments():
    parser_predict = argparse.ArgumentParser(
        description="*** Parse the predicting arguments ***",
    )

    print(parser_predict.description)
    parser_predict.add_argument("input", action="store")
    parser_predict.add_argument("checkpoint", action="store")

    parser_predict.add_argument("--top_k", action="store", dest="top_k", type=int)

    parser_predict.add_argument(
        "--category_names", action="store", dest="category_names"
    )
    parser_predict.add_argument("--gpu", action="store_true", default=False)

    return parser_predict.parse_args()


def prepare_train_validation_data(directory, batch=32):
    train_dir = directory + "/train"
    valid_dir = directory + "/valid"

    train_transforms = transforms.Compose(
        [
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomVerticalFlip(0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    valid_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=batch, shuffle=True
    )

    print(
        f"-> Images in each batch: {batch}\n"
        f"-> Training batches: {len(train_loader)}, images:{len(train_loader) * batch}\n"
        f"-> Validation batches: {len(valid_loader)}, images:{len(valid_loader) * batch}"
    )

    return train_loader, valid_loader, train_data, valid_data


def prepare_test_data(directory, batch=32):
    test_dir = directory + "/test"

    test_transforms = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch, shuffle=True)

    print(
        f"-> Images in each batch: {batch}\n"
        f"-> Test batches: {len(test_loader)}, images:{len(test_loader) * batch}"
    )

    return test_loader


def load_json_file(file):
    with open(file, "r") as f:
        json_info = json.load(f)

    return json_info


def process_image(image):
    # Open an image file
    im = Image.open(image)

    # Scale the image
    size = 256
    width = im.size[0]
    height = im.size[1]
    im.thumbnail((1000, size) if (width > height) else (size, 1000))

    # Crop the image
    left = (im.width - 224) / 2
    right = left + 224
    lower = (im.height - 224) / 2
    upper = lower + 224
    im = im.crop((int(left), int(lower), int(right), int(upper)))

    # Normalize the image
    im = np.array(im) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std

    im = im.transpose((2, 0, 1))

    return im


def imshow(image, ax=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch's tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.show()
    return ax


def predict(image, model, k=3):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """
    im = torch.from_numpy(image).float()

    # From Batch * Channel * Height * Width to 1 picture CxHxW
    im = im.unsqueeze(0)

    model.cpu()
    log = model.forward(im)
    probs = torch.exp(log)
    probs, indices = probs.topk(k)

    classes = []
    indices = indices.detach().numpy()[0]
    for idx in indices:
        for class_, index in model.class_to_idx.items():
            if idx == index:
                classes.append(class_)

    probs = probs.detach().numpy()[0]

    probs_rounded = list(map(lambda x: round(x, 3), probs))

    return probs, probs_rounded, classes


def result(flower_path, probs, probs_rounded, classes, json_file=None, plot=False):
    if json_file:
        cat_to_name = load_json_file(json_file)
        flower_names = []
        for class_ in classes:
            flower_names.append(cat_to_name[class_])
        print(f"-> Flower name: {flower_names}")
        print(f"-> With probabilities: {probs_rounded}")

    else:
        print(f"-> Classes: {classes}")
        print(f"-> With probabilities: {probs_rounded}")

    if plot:
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        fig.suptitle("Flower: names with probabilities between 0 and 1")
        img = plt.imread(flower_path)
        axs[0].imshow(img)

        if json_file:
            bars = axs[1].barh(flower_names, probs_rounded, align="center")
        else:
            bars = axs[1].barh(classes, probs_rounded, align="center")

        axs[1].bar_label(bars)
        plt.show()


# Test
if __name__ == "__main__":
    # path = "/home/naeim/PycharmProjects/PythonAI_Part2/flowers/test/68/image_05927.jpg"
    path = "image_05927.jpg"
    imshow(process_image(path))
