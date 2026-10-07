import requests

# Definir o endereço da nossa API
url = "http://localhost:8000/predict"

# Definir os dados que queremos enviar
dados = {
    "features": [5.1, 3.5, 1.4, 0.2]
}

# Enviar o pedido POST
response = requests.post(url, json=dados)

# Imprimir a resposta da API utilizando a função json()
print(response.json())