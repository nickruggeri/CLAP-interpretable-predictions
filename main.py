import logging
import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path

import torch

from src.architecture.clap import CLAP
from src.data.loading import get_datasets
from src.trainer import CLAPTrainer

if __name__ == "__main__":

    parser = ArgumentParser()
    # experiment
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="MPITOY",
        help="Dataset used for the experiment. Available: MPI, Shapes3D, SmallNORB, PlantVillage, ChestXRay",
    )
    # model
    parser.add_argument(
        "--z_core_dim",
        type=int,
        default=10,
        help="Dimension of the core latent space in the model.",
    )
    parser.add_argument(
        "--z_style_dim",
        type=int,
        default=20,
        help="Dimension of the style latent space in the model.",
    )
    parser.add_argument(
        "--y_reg",
        type=float,
        default=50,
        help="Weight of the prediction error during training.",
    )
    parser.add_argument(
        "--group_sparsity_reg",
        type=float,
        default=0.05,
        help="Weight of the sparsity regularization term during training.",
    )
    # optimization
    parser.add_argument(
        "--lr", type=float, default=1.0e-4, help="Optimizer learning rate."
    )
    parser.add_argument("--batch_size", type=int, default=132, help="Batch size.")
    parser.add_argument(
        "--n_epochs_start",
        type=int,
        default=200,
        help="Initial number of epochs during training. Used for scheduling of the regularization parameters.",
    )
    parser.add_argument(
        "--n_epochs_beta",
        type=int,
        default=400,
        help="Middle number of epochs during training. Used for scheduling of the regularization parameters.",
    )
    parser.add_argument(
        "--n_epochs_end",
        type=int,
        default=200,
        help="Final number of epochs during training. Used for scheduling of the regularization parameters.",
    )
    # logging and saving
    parser.add_argument(
        "--save_path",
        type=Path,
        default=None,
        help="Path of the directory where training results are saved.",
    )
    args = parser.parse_args()

    # set logging and seed
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # load datasets, model, optimizer and trainer
    train_dataset, test_dataset, n_channels, image_dim, n_classes = get_datasets(
        args.dataset
    )
    model = CLAP(n_channels, image_dim, args.z_style_dim, args.z_core_dim, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = CLAPTrainer(
        model, train_dataset, test_dataset, optimizer, args.batch_size, args.save_path
    )

    # save args to pickle for easy restarting of job, reloading of variables or simple check
    if args.save_path is not None:
        pkl_path = args.save_path / "args.pkl"
        with open(pkl_path, "wb") as file:
            pkl.dump(args, file)

    # train
    trainer.train(
        args.y_reg,
        args.group_sparsity_reg,
        1.0,
        args.n_epochs_start,
        args.n_epochs_beta,
        args.n_epochs_end,
    )
