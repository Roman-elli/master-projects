import os
import numpy as np

def readFiles(foldername):
    lista_dados = []

    for arquivo in os.listdir(foldername):
        if arquivo.endswith(".cvg"):
            caminho_arquivo = os.path.join(foldername, arquivo)
            
            try:
                data = np.loadtxt(caminho_arquivo, delimiter=",")
                lista_dados.append(data)
            except Exception as e:
                print(f"Erro lendo {caminho_arquivo}: {e}")

    if lista_dados:
        try:
            return np.array(lista_dados)
        except ValueError:
            return np.array(lista_dados, dtype=object)
    else:
        return np.array([])
