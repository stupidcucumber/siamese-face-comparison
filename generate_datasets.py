import argparse
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    """Parse arguments.

    Returns
    -------
    argparse.Namespace
        Arguments and their values.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root", type=Path, required=True, help="Path to the casia webface dataset."
    )

    parser.add_argument(
        "--train",
        type=float,
        default=0.9,
        help="Portion of the dataset to be reserved for train partition.",
    )

    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the reproducibility."
    )

    return parser.parse_args()


def choose_negative(identities: list[Path]) -> Path:
    """Choose negative image path.

    Parameters
    ----------
    identities : list[Path]
        Folders of identities to choose from.

    Returns
    -------
    Path
        Identity variation considered to be negative.
    """
    identity = random.choice(identities)

    identity_variations = list(identity.glob("*.*"))

    return random.choice(identity_variations)


def generate_training_file(images_directory: Path) -> pd.DataFrame:
    """Generate traininig file.

    Parameters
    ----------
    images_directory : Path
        Path to the root directory with all identities.

    Returns
    -------
    pd.DataFrame
        With three columns:
        - anchor
        - positive
        - negative
    """
    result = []

    identities = list(images_directory.glob("*/"))

    for identity in tqdm(identities, total=len(identities)):

        identity_variations = list(identity.glob("*.jpg"))

        identity_variations.sort(key=lambda x: int(x.stem.split(".")[0]))

        anchor = identity_variations[0]

        positive = identity_variations[1]

        probable_negative_identities = set(identities)

        probable_negative_identities.remove(identity)

        negative = choose_negative(list(probable_negative_identities))

        result.append([anchor, positive, negative])

    return pd.DataFrame(result, columns=["anchor", "positive", "negative"])


def main(root: Path, train: float, seed: int) -> None:
    """Entry the main point of the script.

    Parameters
    ----------
    root : Path
        Path to the root directory of the CASIA-WebFace.
    train : float
        Portion of the dataset to be considered as train.
    seed : int
        Initialize state of the random with seed for reproducibility.
    """
    random.seed(seed)

    training_file = generate_training_file(images_directory=root)

    training_file.to_csv("train_partition.csv")


if __name__ == "__main__":

    args = parse_args()

    try:

        main(**dict(args._get_kwargs()))

    except KeyboardInterrupt:

        print("User interrupted.")
