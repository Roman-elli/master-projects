from utils.io import readFiles
from core.metrics import activity_metric, zscore_outliers
import config as cfg

def main():
    # 2
    data_array = readFiles(cfg.ASSETS_FOLDERS_PATH)

    # 3.1 & 3.2
    modules_acc, activities_acc = activity_metric(data_array, 'acc')
    modules_gyro, activities_gyro = activity_metric(data_array, 'gyro')
    modules_mag, activities_mag = activity_metric(data_array, 'mag')

    # 3.3 & 3.4
    for k_value in [3, 3.5, 4]:
        zscore_outliers(modules_acc, activities_acc, sensor='acc', k=k_value)
        zscore_outliers(modules_gyro, activities_gyro, sensor='gyro', k=k_value)
        zscore_outliers(modules_mag, activities_mag, sensor='mag', k=k_value)

if __name__ == '__main__':
    main()
