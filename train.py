from argparse import ArgumentParser, Namespace
from pathlib import Path
from statistics import mean

import torch
import torchvision
from torch.nn import TripletMarginLoss
from tqdm import tqdm

from src.dataset import SiameeseDataset
from src.metrics import false_accept, false_rate, true_accept, validation_rate
from src.nn import SiameeseNN


def parse_arguments() -> Namespace:
    """Parse arguments from the CLI.

    Returns
    -------
    Namespace
        Arguments extracted from the CLI.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "--train-data",
        type=Path,
        required=True,
        help="Path to the train partition csv file.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=(256, 256),
        help="Size of the image [height, width]",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to train for. By default: 10.",
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size to use during the traininig. By default: 32.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to train on. By default: cuda.",
    )

    return parser.parse_args()


def prepare_embeddings_for_validation(
    anchor_embeddings: torch.Tensor,
    positive_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare embeddings for validation feeding.

    Parameters
    ----------
    anchor_embeddings : torch.Tensor
        Embeddings corresponding to the anchor inputs.
    positive_embeddings : torch.Tensor
        Embeddings corresponding to the positive inputs.
    negative_embeddings : torch.Tensor
        Embeddings corresponding to the negative inputs.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - first part of the pair
        - scond part of the pair
        - labels
    """
    embeddings_i = [anchor_embeddings, positive_embeddings, negative_embeddings]

    embeddings_j = [positive_embeddings, negative_embeddings, anchor_embeddings]

    labels = [
        torch.ones(len(anchor_embeddings), dtype=torch.int32),
        torch.zeros(len(anchor_embeddings), dtype=torch.int32),
        torch.zeros(len(anchor_embeddings), dtype=torch.int32),
    ]

    return (
        torch.concat(embeddings_i, dim=0),
        torch.concat(embeddings_j, dim=0),
        torch.concat(labels, dim=0),
    )


def train(
    model: SiameeseNN,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    train_dataloader: torch.utils.data.DataLoader,
    distance: float,
    device: str,
) -> None:
    """Train neural network with TripletMarginLoss.

    Parameters
    ----------
    model : SiameeseNN
        Target network to train.
    optimizer : torch.optim.Optimizer
        Optimizer to use during traininig.
    epochs : int
        Number of epochs to train for.
    train_dataloader : torch.utils.data.DataLoader
        Dataloader for the train subset.
    distance : float
        Minimum distance to consider pair similar.
    device : str
        Device to optimize on.
    """
    criterion = TripletMarginLoss(margin=0.2)

    for epoch in range(epochs):

        print(f"Epoch {epoch:03d}")

        train_tqdm = tqdm(train_dataloader, total=len(train_dataloader))

        avg_loss = []

        avg_fa = []

        avg_ta = []

        avg_var = []

        avg_far = []

        for anchors, positives, negatives in train_tqdm:

            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)

            anchors_embeddings = model(anchors)

            positive_embeddings = model(positives)

            negative_embeddings = model(negatives)

            optimizer.zero_grad()

            loss: torch.Tensor = criterion(
                anchors_embeddings, positive_embeddings, negative_embeddings
            )

            loss.backward()

            optimizer.step()

            embeddings_i, embeddings_j, labels = prepare_embeddings_for_validation(
                anchors_embeddings, positive_embeddings, negative_embeddings
            )

            ta = true_accept(embeddings_i, embeddings_j, labels, distance)

            fa = false_accept(embeddings_i, embeddings_j, labels, distance)

            var = validation_rate(embeddings_i, embeddings_j, labels, distance)

            far = false_rate(embeddings_i, embeddings_j, labels, distance)

            avg_loss.append(loss.detach().cpu().item())

            avg_ta.append(ta.detach().cpu().item())

            avg_fa.append(fa.detach().cpu().item())

            avg_var.append(var.detach().cpu().item())

            avg_far.append(far.detach().cpu().item())

            train_tqdm.set_description_str(
                f"Training: Loss - {mean(avg_loss):0.3f} TA - {mean(avg_ta):0.3f} "
                f"FA - {mean(avg_fa):0.3f} VAR - {mean(avg_var):0.3f} "
                f"FAR - {mean(avg_far):0.3f}"
            )


def main(
    train_data: Path, imgsz: list[int], epochs: int, batch: int, device: str
) -> None:
    """Entry the main point of the script.

    Parameters
    ----------
    train_data : Path
        Path to the csv file with data for training.
    imgsz : list[int]
        Size of the inputs.
    epochs : int
        Number of epochs to train for.
    batch : int
        Batch size to use for training.
    device : str
        Device to train on.
    """
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
            torchvision.transforms.Resize(size=imgsz),
        ]
    )

    train_dataset = SiameeseDataset(data=train_data, transform=transform)

    model = SiameeseNN()

    model = model.to(device)

    optimizer = torch.optim.SGD(params=model.parameters())

    train(
        model=model,
        optimizer=optimizer,
        epochs=epochs,
        train_dataloader=torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=batch, shuffle=True
        ),
        distance=0.2,
        device=device,
    )


if __name__ == "__main__":

    args = parse_arguments()

    try:

        main(**dict(args._get_kwargs()))

    except KeyboardInterrupt:

        print("User interrupted training.")
