from utils.io import readFiles
from core.metrics import activity_metric, zscore_outliers, k_mean
import config as cfg

def main():
    # 2
    data_array = readFiles(cfg.ASSETS_FOLDERS_PATH)

    # 3.1 & 3.2
    # results_acc = activity_metric(data_array, 'acc', save_dir="data/activity", save_plots=False)
    # results_gyro = activity_metric(data_array, 'gyro', save_dir="data/activity", save_plots=False)
    # results_mag = activity_metric(data_array, 'mag', save_dir="data/activity", save_plots=False)

    # 3.3 & 3.4
    # useActivitie = False
    # for k_value in [3, 3.5, 4]:
    #     zscore_outliers(results_acc, sensor='acc', k=k_value, use_activities=useActivitie, save_dir="data/zscore")
    #     zscore_outliers(results_gyro, sensor='gyro', k=k_value, use_activities=useActivitie, save_dir="data/zscore")
    #     zscore_outliers(results_mag, sensor='mag', k=k_value, use_activities=useActivitie, save_dir="data/zscore")
    
    #3.6
    k_mean(data_array, sensor='mag', n_clusters=cfg.LABELS_COUNTER, save_dir="data/kmean", save_plots=True)
   

if __name__ == '__main__':
    main()
