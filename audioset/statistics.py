import numpy as np
from tqdm import tqdm
import argparse
import pandas as pd
import os
from config import feature_type, num_bins, sample_rate, workspace, use_resampled_data
from utils import getSampleRateString
from glob import glob

__author__ = "Andrew Koh Jin Jie, Yan Zhen"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen",
               "Tanmay Khandelwal", "Anushka Jain"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Soham Tiwari"
__email__ = "soham.tiwari800@gmail.com"
__status__ = "Development"


def run(workspace, feature_type, num_bins, perm):
    if use_resampled_data:
        # file_list = np.unique(glob('{}/data/{}/audio_{}/*.wav.npy'.format(workspace,
        #                                                                   feature_type, getSampleRateString(sample_rate))))
        file_list = np.unique(glob('{}/*.wav.npy'.format(workspace)))
    else:
        #    if '8k' in feature_type:
        #        actual_files = glob('{}/{}/*.wav.npy'.format(workspace, feature_type))
        #        for file in actual_files:
        #            new_filename = file.split('/')[-1].split('.')[0]
        #            os.rename(file, '{}/{}/{}.wav.npy'.format(workspace, feature_type, new_filename))

        # Load and prepare data
        folds = []
        for i in range(5):
            folds.append(pd.read_csv(
                '{}/split/fold_{}_c.txt'.format(workspace, i), delimiter=" ", header=None))

        train_df = pd.concat([folds[perm[0]], folds[perm[1]], folds[perm[2]]])

        file_list = train_df[0].unique()

    mean = np.zeros((num_bins,))
    M2 = np.zeros((num_bins,))

    no_file_count = 0
    n = 0
    for file_name in tqdm(file_list):
        try:
            if use_resampled_data:
                data = np.load(file_name)
            else:
                data = np.load('{}/data/{}/audio_{}/{}.wav.npy'.format(workspace,
                                                                       feature_type, getSampleRateString(sample_rate), file_name))
        except Exception as e:
            no_file_count += 1
            print('{}/data/{}/audio_{}/{}.wav.npy'.format(workspace,
                                                          feature_type, getSampleRateString(sample_rate), file_name))
            print(e)
            continue
        x = data.mean(axis=1)
        n += 1
        delta = x - mean
        mean = mean + delta/n
        M2 = M2 + delta*(x - mean)

    variance = M2/(n - 1)
    stdev = np.sqrt(variance)
    print(no_file_count, n)

    folder_path = '{}/data/statistics/{}'.format(
        workspace, getSampleRateString(sample_rate))
    os.makedirs(folder_path, exist_ok=True)
    np.save('{}/channel_means_{}_{}'.format(folder_path,
            feature_type, str(perm[0])+str(perm[1])+str(perm[2])), mean)
    np.save('{}/channel_stds_{}_{}'.format(folder_path,
            feature_type, str(perm[0])+str(perm[1])+str(perm[2])), stdev)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Feature type')
    parser.add_argument('-w', '--workspace', type=str, default=workspace)
    # parser.add_argument('-d', '--dataset', type=str)
    parser.add_argument('-f', '--feature_type', type=str, default=feature_type)
    parser.add_argument('-n', '--num_bins', type=int, default=num_bins)
    parser.add_argument('-p', '--permutation', type=int,
                        nargs='+', required=True)
    args = parser.parse_args()
    run(args.workspace, args.feature_type, args.num_bins, args.permutation)
