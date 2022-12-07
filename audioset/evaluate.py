import enum
from matplotlib import use
import pandas as pd
import os
import sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score
import argparse
from utils import AudioDataset, Task5Model, configureTorchDevice, getSampleRateString
from config import target_names, feature_type, num_frames, permutation, batch_size, num_workers, num_classes, sample_rate, workspace, use_cbam, seed, use_resampled_data
from glob import glob

__author__ = "Andrew Koh Jin Jie, Anushka Jain and Soham Tiwari"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Soham Tiwari"
__email__ = "soham.tiwari800@gmail.com"
__status__ = "Development"

class_mapping = {}
for i, target in enumerate(target_names):
    class_mapping[target] = i


def run(workspace, feature_type, num_frames, perm, model_arch, use_cbam, expt_name):

    if use_resampled_data:
        # file_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/data/{}/audio_{}/*.wav.npy'.format(workspace,
        #                                                                                                      feature_type, getSampleRateString(sample_rate))))]
        # file_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/*.wav.npy'.format(workspace,
        #                                                                                                 feature_type, getSampleRateString(sample_rate))))]
        train_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/train_labels/*.wav.npy'.format(workspace,
                                                                                                        feature_type, getSampleRateString(sample_rate))))]    
        val_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/val_labels/*.wav.npy'.format(workspace,
                                                                                                feature_type, getSampleRateString(sample_rate))))]
        test_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/test_labels/*.wav.npy'.format(workspace,
                                                                                                        feature_type, getSampleRateString(sample_rate))))]  
        # train_list, test_list = sklearn.model_selection.train_test_split(
        #     file_list, train_size=0.8, random_state=seed)
        # train_list, val_list = sklearn.model_selection.train_test_split(
        #     train_list, train_size=0.9, random_state=seed)
        train_df = pd.DataFrame(train_list)
        valid_df = pd.DataFrame(val_list)
        test_df = pd.DataFrame(test_list)
    else:
        folds = []
        for i in range(5):
            folds.append(pd.read_csv(
                '{}/split/fold_{}_c.txt'.format(workspace, i), delimiter=" ", header=None))

        train_df = pd.concat([folds[perm[0]], folds[perm[1]], folds[perm[2]]])
        valid_df = folds[perm[3]]
        test_df = folds[perm[4]]

    # Create the datasets and the dataloaders
    train_dataset = AudioDataset(
        workspace, train_df, feature_type=feature_type, perm=perm, resize=num_frames, usage='train')
    valid_dataset = AudioDataset(
        workspace, valid_df, feature_type=feature_type, perm=perm, resize=num_frames, usage='val')
    test_dataset = AudioDataset(
        workspace, test_df, feature_type=feature_type, perm=perm, resize=num_frames, usage='test')
    test_loader = DataLoader(test_dataset, batch_size,
                             shuffle=False, num_workers=num_workers)
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    device = configureTorchDevice()

    # Instantiate the model
    model = Task5Model(num_classes, model_arch, use_cbam=use_cbam).to(device)
    model_path = '{}/model/{}/{}/model_{}_{}_{}_use_cbam_{}'.format(workspace, expt_name, getSampleRateString(
        sample_rate), feature_type, str(perm[0])+str(perm[1])+str(perm[2]), model_arch, use_cbam)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    print(f'Using {model_arch} model from {model_path}.')

    y_pred = []
    for sample in test_loader:
        inputs = sample['data'].to(device)
        labels = sample['labels'].to(device)

        with torch.set_grad_enabled(False):
            model = model.eval()
            outputs = model(inputs)
            for i in range(len(outputs)):
                curr = outputs[i]
                arg = torch.argmax(curr)
                y_pred.append(arg.detach().cpu().numpy())
    y_true = []

    for index, row in test_df.iterrows():
        class_name = row[0].split('-')[0]
        y_true.append(class_mapping[class_name])

    print(f'Including other class:')
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Micro F1 Score: {f1_score(y_true, y_pred, average='micro')}")
    print(f"Macro F1 Score: {f1_score(y_true, y_pred, average='macro')}")
    print(f'Accuracy Score: {accuracy_score(y_true, y_pred)}')
    print(y_true[:5], y_pred[:5])

    y_true_new = []
    y_pred_new = []

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt != 9:
            y_true_new.append(yt)
            y_pred_new.append(yp)

    print(len(y_true), len(y_true_new), np.unique(
        y_true_new, return_counts=True))
    print(f'Excluding other class:')
    print(classification_report(y_true_new, y_pred_new, digits=4))
    print(
        f"Micro F1 Score: {f1_score(y_true_new, y_pred_new, average='micro')}")
    print(
        f"Macro F1 Score: {f1_score(y_true_new, y_pred_new, average='macro')}")
    print(f'Accuracy Score: {accuracy_score(y_true_new, y_pred_new)}')
    print(y_true_new[:5], y_pred_new[:5])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Feature type')
    parser.add_argument('-en', '--expt_name', type=str, required=True)
    parser.add_argument('-w', '--workspace', type=str, default=workspace)
    parser.add_argument('-f', '--feature_type', type=str, default=feature_type)
    parser.add_argument('-n', '--num_frames', type=int, default=num_frames)
    parser.add_argument('-ma', '--model_arch', type=str, default='mobilenetv2')
    parser.add_argument('-cbam', '--use_cbam', action='store_true')
    parser.add_argument('-p', '--permutation', type=int,
                        nargs='+', default=permutation)
    args = parser.parse_args()
    run(args.workspace, args.feature_type, args.num_frames,
        args.permutation, args.model_arch, args.use_cbam, args.expt_name)
