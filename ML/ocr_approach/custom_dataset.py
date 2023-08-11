from torch.utils.data import Dataset
import pandas as pd 


class CustomOCRDataset(Dataset):
    def __init__(self, annotations_file):
        self.ocr_labels = pd.read_csv(annotations_file)
        
    def __len__(self):
        return len(self.ocr_labels)

    def __getitem__(self, idx):
        ocr_path = self.ocr_labels.iloc[idx, 1]
        with open(ocr_path, 'r') as file:
            ocr_data = file.read()
        label = self.ocr_labels.iloc[idx, 0]
        return label, ocr_data