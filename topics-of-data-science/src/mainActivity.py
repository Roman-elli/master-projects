from utils.io import readFiles, outliers
import config as cfg

def main():
    # Exercise 2
    data_array = readFiles(cfg.ASSETS_FOLDERS_PATH)
    analyzed_outliers = outliers(data_array)
    

if __name__ == '__main__':
    main()