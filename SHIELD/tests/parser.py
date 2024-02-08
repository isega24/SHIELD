import argparse


def arg_parser():
    parser = argparse.ArgumentParser(description="")
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
        help="Force to overwrite previous results",
    )
    parser.add_argument(
        "--dataset", metavar="D", type=str, default="CIFAR10", help="Dataset"
    )
    parser.add_argument(
        "--lr", metavar="L", type=float, default=4e-5, help="Learning rate"
    )
    parser.add_argument(
        "--shield", action="store_true", help="Constraint of taking random features"
    )
    parser.add_argument(
        "--xshield", action="store_true", help="Constraint of taking low important features"
    )
    parser.add_argument(
        "--batch_size", metavar="B", type=int, default=32, help="Batch size"
    )
    parser.add_argument(
        "--pretrained_model",
        metavar="P",
        type=str,
        default="efficientnet-b2",
        help="Pretrained model",
    )
    parser.add_argument(
        "--percentage",
        metavar="P",
        type=float,
        default=0,
        help="Percentage of perturbation from 0 to 100",
    )
    return parser
