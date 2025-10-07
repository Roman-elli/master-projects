from utils.io import readFiles
from core.outlier import outliers

import config as cfg

def main():
    # 2
    data_array = readFiles(cfg.ASSETS_FOLDERS_PATH)

    # 3.1
    outliers(data_array)
    
if __name__ == '__main__':
    main()