import torch

torch.set_num_threads(1)
from torch.utils.data.dataloader import DataLoader
import tqdm
import numpy as np
import json
import argparse
import pandas as pd
import os

from ReVel.LLEs import get_xai_model
from ReVel.perturbations import get_perturbation
from ReVel.load_data import load_data
from ReVel.revel.revel import ReVel
import SHIELD.procedures as procedures

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"  # Forcing to use CPU
# fijar la semillas
torch.manual_seed(3141516)
np.random.seed(3141516)

n_classes = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "EMNIST": 47,
    "FashionMNIST": 10,
    "Flowers": 102,
    "OxfordIIITPet": 37,
}

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument(
    "--run_fold",
    metavar="F",
    type=str,
    default="runs/Flowers/efficientnet_v2_s_baseline_0.00/",
)

args = parser.parse_args()

with open(args.run_fold + "config.json") as f:
    config = json.load(f)

DATASET = config.get("dataset")
lr = config.get("lr")

name = "metrics.csv"

n_class = n_classes[DATASET]
iterations = 10
batch_size = 32
max_examples = 1000
samples = 500
sigma = 12
xai_model = "LIME"
dim = 8

csv_file = args.run_fold + name

if os.path.exists(csv_file):
    os.remove(csv_file)


perturbation = get_perturbation(
    name="square", dim=dim, num_classes=n_class, final_size=(224, 224)
)
Test = load_data(DATASET, perturbation=perturbation, train=False, dir="./data")
# Hacer que Test tenga solo las primeras 'samples' de Test
if isinstance(samples, int):
    indices = np.random.choice(
        [i for i in range(len(Test))], size=samples, replace=False
    )
    Test = torch.utils.data.Subset(Test, indices)
TestLoader = iter(DataLoader(Test, batch_size=1, shuffle=False))
classifier = procedures.classifier(config.get("pretrained_model"), n_classes[DATASET])
classifier.to(device)
state_dict = torch.load(f"{args.run_fold}/model.pt", map_location=device)
classifier.load_state_dict(state_dict)
classifier.to(device)
classifier.eval()
print("Loaded the pretrained model.")
total = 0
experiments_done = []
if os.path.exists(csv_file):
    bigDF = pd.read_csv(csv_file)
    experiments_done = np.unique(bigDF["index"])


for data in tqdm.tqdm(TestLoader, total=len(TestLoader)):
    inputs, labels = data
    for k, inp in enumerate(inputs):
        if not (total in experiments_done):
            inp = inp.to(device)

            # inp dims: (C,H,W) -> (H,W,C)
            inp = np.transpose(inp, (1, 2, 0))

            labels = labels[k].to(device)
            explainer = get_xai_model(
                name=xai_model,
                perturbation=perturbation,
                max_examples=max_examples,
                dim=dim,
                sigma=sigma,
            )

            def classify(image, model=classifier):
                """
                This function takes an image and returns the predicted probabilities.
                :param image: A tensor of shape HxWxC
                :return: A tensor of shape Cx1
                """
                if isinstance(image, np.ndarray):
                    image = np.expand_dims(image, 0)

                    image = torch.Tensor(image).to(device)

                else:
                    image = torch.unsqueeze(image, 0)

                # image dims: (N,H,W,C) -> (N,C,H,W)

                image = torch.transpose(image, 3, 2).transpose(2, 1)

                result = model(image)
                return result

            def model_fordward(
                X: np.array, explainator=explainer, model=classify, img=inp
            ):
                """
                This function takes a feature vector and returns the predicted probabilities of the original img.
                :param X: A tensor of shape F.
                :param explainator: An explainator object.
                :param model: A function that takes an image and returns the predicted probabilities.
                    This function accept an image of shape HxWxC and returns a tensor of shape Cx1.
                :param img: The original image.
                :return: A tensor of shape Cx1
                """

                neutral = explainator.perturbation.fn_neutral_image(img)

                avoid = [i for i in range(len(X)) if X[i] == 0]

                segments = explainator.perturbation.segmentation_fn(img.numpy())
                perturbation = explainator.perturbation.perturbation(
                    img, neutral, segments=segments, indexes=avoid
                )
                return model(perturbation)

            segments = explainer.perturbation.segmentation_fn(inp.numpy())

            revel = ReVel(
                model_f=classify,
                model_g=model_fordward,
                instance=inp,
                lle=explainer,
                n_classes=n_class,
                segments=segments,
            )
            df = revel.evaluate(times=iterations)

            # Aniade los argumentos como columnas de la tabla df
            df.loc[:, "dataset"] = DATASET
            df.loc[:, "name"] = name
            df.loc[:, "index"] = total

            if os.path.exists(csv_file):
                bigDF = pd.read_csv(csv_file)
                bigDF = pd.concat([bigDF, df])
            else:
                bigDF = df
            bigDF.to_csv(csv_file, index=False)

        total += 1
