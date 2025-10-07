import os
import numpy as np
import matplotlib as mp

def readFiles(foldername):
    data = []

    for folder in os.listdir(foldername):
        folder_path = os.path.join(foldername, folder)
        
        data_individual = []

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            array = np.loadtxt(file_path, delimiter=',')
            data_individual.append(array)

        if data_individual:
            data.append(data_individual)

    if not data:
        print("File not found")
    return data

