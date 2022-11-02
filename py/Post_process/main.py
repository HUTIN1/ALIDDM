import csv
import pandas as pd
import os
import argparse
from tqdm import tqdm


def main(args):
    df_train = pd.read_csv(os.path.join(args.mount_point, args.csv_train))
    df_val = pd.read_csv(os.path.join(args.mount_point, args.csv_valid))
    df_test = pd.read_csv(os.path.join(args.mount_point, args.csv_valid))

    for idx in tqdm(range(len(df_train)), total=len(df_train)):
        os.system('python post_process.py --surf '+args.mount_point+'/'+df_train.iloc[idx]["surf"]+" --remove_islands True --out "+args.mount_point_out+'/'+df_train.iloc[idx]["surf"])





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Post process')

    parser.add_argument('--csv_train', help='CSV with column surf', type=str, required=True)    

    parser.add_argument('--csv_valid', help='CSV with column surf', type=str, required=True)
    parser.add_argument('--csv_test', help='CSV with column surf', type=str, required=True)
    parser.add_argument('--mount_point', help='Dataset mount directory', type=str, default="/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth")


    parser.add_argument('--mount_point_out',help='Dataset mount directory out', type=str, default="/home/luciacev/Desktop/Data/Flybycnn/SegmentationTeeth/postprocess")



    args = parser.parse_args()

    main(args)

