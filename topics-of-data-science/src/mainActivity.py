from utils.io import readFiles
from core.metrics import activity_metric, zscore_outliers
import config as cfg
from sklearn.cluster import KMeans

def main():
    # 2
    data_array = readFiles(cfg.ASSETS_FOLDERS_PATH)

    # 3.1 & 3.2
    results_acc = activity_metric(data_array, 'acc')
    results_gyro = activity_metric(data_array, 'gyro')
    results_mag = activity_metric(data_array, 'mag')

    # 3.3 & 3.4
    for k_value in [3, 3.5, 4]:
        zscore_outliers(results_acc, sensor='acc', k=k_value)
        zscore_outliers(results_gyro, sensor='gyro', k=k_value)
        zscore_outliers(results_mag, sensor='mag', k=k_value)
    
    #3.6
    

if __name__ == '__main__':
    main()
