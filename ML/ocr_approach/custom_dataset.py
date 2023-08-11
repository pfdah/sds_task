from torch.utils.data import Dataset
import pandas as pd 


class CustomOCRDataset(Dataset):
    """Custom Dataset for the task

    Parameters
    ----------
    Dataset : Dataset
        Inheritance of the Torch's Dataset Class for our Custom Dataset class
    """
    def __init__(self, annotations_file: str):
        """Initialize the dataset

        Parameters
        ----------
        annotations_file : str
            The path for CSV with label, OCR-file-path pair.
        """
        # Reads the csv file as the dataset object is created
        self.ocr_labels = pd.read_csv(annotations_file)
        
    def __len__(self):
        """Helper function to return the length of dataset

        Returns
        -------
        int
            the length of dataset
        """
        return len(self.ocr_labels)

    def __getitem__(self, idx:int):
        """Returns the item at the index

        Parameters
        ----------
        idx : int
            the requested index

        Returns
        -------
        CustomOCRDataset
            An instance of the dataset
        """
        # Reads the csv file and sets the label and the ocr_data from the file path of the CSV
        ocr_path = self.ocr_labels.iloc[idx, 1]
        with open(ocr_path, 'r') as file:
            ocr_data = file.read()
        label = self.ocr_labels.iloc[idx, 0]
        return label, ocr_data