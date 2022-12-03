"""
* The purpose of this module is:
    * to read command line arguments,
    * to load data and
    * to preprocess the images
"""
import json
import torch
import argparse
from torchvision import datasets, transforms


def parse_train_arguments():
    parser_train = argparse.ArgumentParser(
        description="*** Parse the training arguments ***",
    )

    print(parser_train.description)
    parser_train.add_argument("data_dir", action="store")

    # use -- to define optional arguments
    parser_train.add_argument("--save_dir", action="store", dest="save_dir")
    parser_train.add_argument("--arch", action="store", dest="arch")
    parser_train.add_argument("--learning_rate", action="store", dest="learning_rate")
    parser_train.add_argument(
        "--hidden_units", action="store", dest="hidden_units", type=int
    )
    parser_train.add_argument("--gpu", action="store_true", default=False)
    parser_train.add_argument("--epochs", action="store", dest="epochs", type=int)

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

    return train_loader, valid_loader


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
