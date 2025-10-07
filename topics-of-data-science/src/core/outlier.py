import matplotlib.pyplot as plt
import numpy as np
import random

def outliers(data_array):
    for individuo_idx, individuo in enumerate(data_array):
        plt.figure(figsize=(15, 10))
        plt.suptitle(f"Indivíduo {individuo_idx + 1}", fontsize=16)

        for i in range(5):
            activity_idx = random.randint(0, len(individuo) - 1)
            activity = individuo[activity_idx]

            sensor_type = random.choice(['acc', 'gyro', 'mag'])

            if sensor_type == 'acc':
                cols = [2, 3, 4]
                label = 'Accelerometer'
            elif sensor_type == 'gyro':
                cols = [5, 6, 7]
                label = 'Gyroscope'
            else:
                cols = [8, 9, 10]
                label = 'Magnetometer'

            x = activity[:, cols[0]]
            y = activity[:, cols[1]]
            z = activity[:, cols[2]]
            module = np.sqrt(x**2 + y**2 + z**2)

            col12 = activity[:, 11]

            plt.subplot(2, 3, i + 1)
            plt.scatter(module, col12)
            plt.plot(col12, module)
            plt.title(f"Atividade {activity_idx + 1} ({label})")
            plt.xlabel("Coluna 12")
            plt.ylabel("Módulo do vetor")

        plt.tight_layout()
        plt.show()
