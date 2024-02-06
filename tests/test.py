import argparse
import torch

torch.set_num_threads(3)
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from ReVel.perturbations import get_perturbation
from ReVel.load_data import load_data
import json
from torch.utils.tensorboard.writer import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from SHIELD.procedures import procedures

import os

os.environ["MPLCONFIGDIR"] = os.getcwd() + "/configs/"

# Script que calcula el accuracy y el loss de un modelo entrenado
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--saved_dir", metavar="F", type=str)
    parser.add_argument(
        "--seed",
        metavar="S",
        type=int,
        default=3141516,
        help="Seed for random number generator",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force to overwrite the saved test accuracy",
    )
    args = parser.parse_args()
    event_acc = EventAccumulator(f"{args.saved_dir}/")
    event_acc.Reload()
    if (not args.force) and ("Test/Accuracy" in event_acc.Tags()["scalars"]):
        print("Test accuracy already computed. Use --force to overwrite")
        exit(0)

    torch.random.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_classes = {
        "CIFAR10": 10,
        "CIFAR100": 100,
        "EMNIST": 47,
        "FashionMNIST": 10,
        "StandfordCars": 196,
        "Flowers": 102,
        "FGVC": 102,
        "OxfordIIITPet": 37,
        "Food101": 101,
        "ImageNet": 1000,
    }

    config = json.load(open(args.saved_dir + "/config.json"))

    dataset = config.get("dataset")

    perturbation = get_perturbation(
        name="square",
        dim=9,
        num_classes=n_classes[dataset],
        final_size=(224, 224),
        kernel=150.0,
        max_dist=20,
        ratio=0.5,
    )

    Test = load_data(dataset, perturbation=perturbation, train=False, dir="./data")

    num_classes = n_classes[dataset]

    pretrained_model = config.get("pretrained_model")
    save_model_dir = f"{args.saved_dir}/model.pt"

    classifier = procedures.classifier(config.get("pretrained_model"), n_classes[dataset])
    classifier.to(device)

    classifier.load_state_dict(torch.load(save_model_dir, map_location=device))

    batch_size = 8
    TestLoader = DataLoader(Test, batch_size=batch_size, shuffle=False)


    def loss_f(ypred, y_label):
        return F.cross_entropy(ypred, torch.argmax(y_label, 1))

    optimizer = torch.optim.AdamW(
        classifier.parameters(), lr=config["lr"], weight_decay=0.01, amsgrad=True
    )

    best_loss = float("inf")

    # Que writer incluya el archivo json de configuración en el directorio

    acc = 0
    loss = 0
    total = 0

    classifier = classifier.eval().to(device)
    with torch.no_grad():
        for data in TestLoader:
            inputs, labels = data

            inputs = inputs.float().to(device)
            labels = labels.to(device)

            outputs = classifier(inputs)

            labelsAcc = torch.argmax(labels, axis=-1)
            outputsAcc = torch.argmax(outputs, axis=-1)

            result = (outputsAcc == labelsAcc).float()

            acc += torch.sum(result)

            loss += loss_f(outputs, labels)
            total += len(inputs)

    writer = SummaryWriter(log_dir=args.saved_dir)
    print(f"Accuracy: {acc / total}")
    print(f"Loss: {loss / total}")
    # Aniade en el writer el accuracy y el loss de test
    writer.add_scalar("Test/Accuracy", acc / total, 0)
    writer.add_scalar("Test/Loss", loss / total, 0)
