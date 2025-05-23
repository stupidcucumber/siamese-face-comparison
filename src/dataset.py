from torch.utils.data import Dataset
import torch
import cv2
from pathlib import Path
import pandas as pd


class SiameeseDataset(Dataset):
    def __init__(self, data: Path, transform) -> None:
        super(SiameeseDataset, self).__init__()
        
        self.data_path = data
        self.transform = transform
        self.data = pd.read_csv(data, index_col=0)
        
    def __len__(self) -> int:
        """Get length of the dataset.
        
        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.data)
    
    def _preprocess_identity(self, image_path: Path) -> torch.Tensor:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.transform(image)
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get item from the dataset.

        Parameters
        ----------
        index : int
            Index of the element to extract.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - First image
            - Second image
            - Third image
        """
        identities = self.data.iloc[index]
        
        anchor_identity_path = identities["anchor"]
        
        positive_identity_path = identities["positive"]
        
        negative_identity_path = identities["negative"]
        
        return (
            self._preprocess_identity(anchor_identity_path),
            self._preprocess_identity(positive_identity_path),
            self._preprocess_identity(negative_identity_path)
        )