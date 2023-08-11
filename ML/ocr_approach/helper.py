import os
import csv

def create_data_csv(path: str):
    """Create the first CSV

    Parameters
    ----------
    path : str
        the path of where the root folder of dataset is present
    """
    # Creates a list of dictionary for the each row of the csv file
    file_label_pair= []
    for label in os.listdir(os.path.join(path,'ocr')):
        for file in os.listdir(os.path.join(path,'ocr',label)):
            file_label_pair.append({'label':label,'file':'../data/ocr/'+label+'/'+ file})
    
    keys = file_label_pair[0].keys()
    with open('dataset.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(file_label_pair)

def map_values(value:int):
    """Map the value of labels from 0 to 4

    Parameters
    ----------
    value : int
        Unmapped value

    Returns
    -------
    int
        the mapped value
    """
    if value == 0:
        return 0
    elif value == 2:
        return 1
    elif value == 4:
        return 2
    elif value == 6:
        return 3
    elif value == 9:
        return 4
    

    
