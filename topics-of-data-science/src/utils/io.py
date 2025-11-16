import os
import numpy as np

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

def read_data_per_person(foldername):
    data_per_person = []

    for idx, person_folder in enumerate(os.listdir(foldername), start=1):
        person_path = os.path.join(foldername, person_folder)
        if not os.path.isdir(person_path):
            continue

        person_data_list = []

        for file in os.listdir(person_path):
            file_path = os.path.join(person_path, file)
            array = np.loadtxt(file_path, delimiter=',')
            person_data_list.append(array)
        
        if person_data_list:
            person_data = np.vstack(person_data_list)
            id_column = np.full((person_data.shape[0], 1), idx)
            person_data = np.hstack((person_data, id_column))
            
            data_per_person.append(person_data)

    return data_per_person

