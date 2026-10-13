
"""
Main command-line interface for the brain tumor project.

Commands:
    train
    evaluate
    predict
"""

import argparse

from src.train import train_model

from src.evaluate import evaluate_model

from src.predict import predict_image


def main():

    parser = argparse.ArgumentParser(

        description=(

            "AI-Powered Medical Image Analysis "

            "- Brain Tumor Classification"

        )

    )

    subparsers = parser.add_subparsers(

        dest="command",

        required=True

    )

    # Training command.

    train_parser = subparsers.add_parser(

        "train",

        help="Train the model"

    )

    train_parser.add_argument(

        "--epochs",

        type=int,

        default=5

    )

    # Evaluation command.

    subparsers.add_parser(

        "evaluate",

        help="Evaluate the trained model"

    )

    # Prediction command.

    predict_parser = subparsers.add_parser(

        "predict",

        help="Predict an MRI image"

    )

    predict_parser.add_argument(

        "--image",

        required=True

    )

    args = parser.parse_args()

    if args.command == "train":

        train_model(

            epochs=args.epochs

        )

    elif args.command == "evaluate":

        evaluate_model()

    elif args.command == "predict":

        predict_image(

            args.image

        )


if __name__ == "__main__":

    main()
