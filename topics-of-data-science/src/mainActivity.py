from utils.io import readFiles
from core.metrics import activity_metric, zscore_outliers

import config as cfg

def main():
    # 2
    data_array = readFiles(cfg.ASSETS_FOLDERS_PATH)

    # 3.1 & 3.2
    #activity_metric(data_array, 'acc')
    #activity_metric(data_array, 'gyro')
    #activity_metric(data_array, 'mag')

    # 3.3 & 3.4
    for k_value in [3, 3.5, 4]:
        zscore_outliers(data_array, sensor='acc', k=k_value)
        zscore_outliers(data_array, sensor='gyro', k=k_value)
        zscore_outliers(data_array, sensor='mag', k=k_value)
    
if __name__ == '__main__':
    main()