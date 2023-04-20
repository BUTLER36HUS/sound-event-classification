import enum
from matplotlib import use
import pandas as pd
import os
import sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, accuracy_score
import argparse
from utils import AudioDataset, Task5Model, configureTorchDevice, getSampleRateString, BalancedBatchSampler,AudioWavDataset
from config import target_names, class_mapping, feature_type, num_frames, permutation, batch_size, num_workers, num_classes, sample_rate, workspace, use_cbam, \
                        seed, use_resampled_data,hop_length,use_raw_wav
from glob import glob
from pathlib import Path

__author__ = "Andrew Koh Jin Jie, Anushka Jain and Soham Tiwari"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Soham Tiwari"
__email__ = "soham.tiwari800@gmail.com"
__status__ = "Development"

# class_mapping = {}
# for i, target in enumerate(target_names):
#     class_mapping[target] = i


"""
Post process the results.
For example, merge certain labels like (Female Speech and Male Speech)

Params:
    arg: prediction from model
    label_change: a dictionary, key should be the original label, value should be the post-merged label
"""
def postprocess(arg,label_change=dict()):
    arg = arg.detach().cpu().numpy()
    # change label
    if arg.item() in label_change:
        arg = label_change[arg.item()]
    # for pre_label,post_label in label_change.items():
    #     print(pre_label,post_label,"\n",arg)
    #     arg[np.where(arg==pre_label)[0]]=post_label
    return arg

def run(workspace, feature_type, num_frames, perm, model_arch, use_cbam, expt_name, args):

    if use_resampled_data:

        removed = set()
        with open("remove_test.txt") as f:
            for i in f.readlines():
                removed.add(i.strip()[:-4])
        ignored_labels = args.ignored_labels
        for label in ignored_labels:
            if label in class_mapping:
                del class_mapping[label]
        num_classes = max(class_mapping.values())+1
        file_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/data/{}/audio_{}/*.wav.npy'.format(workspace,
                                                                                                             feature_type, getSampleRateString(sample_rate))))]
        # train_list, test_list = sklearn.model_selection.train_test_split(
        #     file_list, train_size=0.8, random_state=seed)
        # train_list, val_list = sklearn.model_selection.train_test_split(
        #     train_list, train_size=0.9, random_state=seed)

        # file_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/data/{}/audio_{}/*.wav.npy'.format(workspace,
        #                                                                                                      feature_type, getSampleRateString(sample_rate))))]
        # file_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/*.wav.npy'.format(workspace,
        #                                                                                                 feature_type, getSampleRateString(sample_rate))))]
        train_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/train_logmel/{}/*.wav.npy'.format(workspace, f'sr={sample_rate}_hop={hop_length}',
                                                                                                        feature_type, getSampleRateString(sample_rate))))  if os.path.basename(p).split('-')[0] not in ignored_labels]    
        val_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/val_logmel/{}/*.wav.npy'.format(workspace, f'sr={sample_rate}_hop={hop_length}',
                                                                                                feature_type, getSampleRateString(sample_rate))))  if os.path.basename(p).split('-')[0] not in ignored_labels]
        test_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/test_logmel/{}/*.wav.npy'.format(workspace, f'sr={sample_rate}_hop={hop_length}',
                                                                                                        feature_type, getSampleRateString(sample_rate))))  if os.path.basename(p).split('-')[0] not in ignored_labels]  
        # train_list, test_list = sklearn.model_selection.train_test_split(
        #     file_list, train_size=0.8, random_state=seed)
        # train_list, val_list = sklearn.model_selection.train_test_split(
        #     train_list, train_size=0.9, random_state=seed)
        # print(len(train_list), len(val_list), len(test_list))

        train_df = pd.DataFrame(train_list)
        valid_df = pd.DataFrame(val_list)
        test_df = pd.DataFrame(test_list)
        # test_df = valid_df
    elif use_raw_wav:
        removed = set()
        with open("remove_test.txt") as f:
            for i in f.readlines():
                removed.add(i.strip()[:-4])
        ignored_labels = args.ignored_labels
        for label in ignored_labels:
            if label in class_mapping:
                del class_mapping[label]
        num_classes = max(class_mapping.values())+1
        # file_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/data/{}/audio_{}/*.wav.npy'.format(workspace,
        #                                                                                                      feature_type, getSampleRateString(sample_rate))))]
        train_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/train_embed/*.wav.npy'.format(workspace)))]    
        val_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/val_embed/*.wav.npy'.format(workspace)))]
        test_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/test_embed/*.wav.npy'.format(workspace)))]  
        # train_list, test_list = sklearn.model_selection.train_test_split(
        #     file_list, train_size=0.8, random_state=seed)
        # train_list, val_list = sklearn.model_selection.train_test_split(
        #     train_list, train_size=0.9, random_state=seed)
        print(len(train_list), len(val_list), len(test_list))

        train_df = pd.DataFrame(train_list)
        valid_df = pd.DataFrame(val_list)
        test_df = pd.DataFrame(test_list)
    else:
        folds = []
        for i in range(5):
            folds.append(pd.read_csv(
                '{}/model/split/fold_{}_c.txt'.format(workspace, i), delimiter=" ", header=None))

        train_df = pd.concat([folds[perm[0]], folds[perm[1]], folds[perm[2]]])
        valid_df = folds[perm[3]]
        test_df = folds[perm[4]]
        # test_df = valid_df
    print(f"args.gAll={args.gaussian_all} args.pnotes={args.process_notes}")
    print(args)
    if args.use_val: test_df = valid_df
    spec_transforms = transforms.Compose([
        transforms.GaussianBlur((5,5)),
    ])

    # Create the datasets and the dataloaders
    # train_dataset = AudioDataset(
    #     workspace, train_df, feature_type=feature_type, perm=perm, resize=num_frames, usage='train'
    if use_raw_wav:
        # valid_dataset = AudioWavDataset(
        #     workspace, valid_df, usage='val')
        test_dataset = AudioWavDataset(
            workspace, test_df,  usage='test')
    else:
        valid_dataset = AudioDataset(
            workspace, valid_df, feature_type=feature_type, perm=perm, resize=num_frames, usage='val',  spec_transform=spec_transforms if args.gaussian_all else None)
        test_dataset = AudioDataset(
            workspace, test_df, feature_type=feature_type, perm=perm, resize=num_frames, usage='test', spec_transform=spec_transforms if args.gaussian_all else None )
    test_loader = DataLoader(test_dataset, batch_size,
                             shuffle=False, num_workers=num_workers)
    if args.use_val:
        test_loader = DataLoader(valid_dataset, batch_size,
                                shuffle=False, num_workers=num_workers)
    # print(len(train_dataset), len(valid_dataset), len(test_dataset))
    device = configureTorchDevice()

    
    print(class_mapping)
    idx_to_class = dict((v,k) for k,v in class_mapping.items())
    # Instantiate the model
    
    model = Task5Model(num_classes, model_arch, use_cbam=use_cbam, dataset=None, dataset_sampler=BalancedBatchSampler(train_df),knotes=args.knotes,process_notes=args.process_notes,dropout=args.dropout,double_mlp = args.double_mlp, layer_pooled = args.layer_pooled,linear_merge = args.linear_merge).to(device)
    
    model_path = '{}/model/{}/{}/model_{}_{}_{}_use_cbam_{}{}'.format(workspace, expt_name, getSampleRateString(
        sample_rate), feature_type, str(perm[0])+str(perm[1])+str(perm[2]), model_arch, use_cbam,args.model_suffix)
    model.load_state_dict(torch.load(model_path,map_location=device)['model_state_dict'])
    model.to(device)
    print(f'Using {model_arch} model from {model_path}.')

    confusion_matrix = np.zeros((num_classes, num_classes))

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
                if arg!=labels[i]:
                    pass 
                arg = postprocess(
                    arg,)
                confusion_matrix[labels[i]][arg] += 1
                y_pred.append(arg)
    y_true = []


    for index, row in test_df.iterrows():
        class_name = row[0].split('-')[0]
        y_true.append(class_mapping[class_name])


    print(f'Including other class:')
    print(classification_report(y_true, y_pred, digits=4))
    print("Micro F1 Score: {:.2f}".format(f1_score(y_true, y_pred, average='micro')))
    print("Macro F1 Score: {:.2f}".format(f1_score(y_true, y_pred, average='macro')))
    print('Accuracy Score: {:.2f}'.format(accuracy_score(y_true, y_pred)))
    print(y_true[:5], y_pred[:5])
    np.save('{}/model/{}/{}/confusion_matrix_{}_{}_{}_use_cbam_{}'.format(workspace, expt_name, getSampleRateString(
        sample_rate), feature_type, str(perm[0])+str(perm[1])+str(perm[2]), model_arch, use_cbam), confusion_matrix)
    
    return 

    y_true_new = []
    y_pred_new = []

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt != class_mapping['others']:
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
    np.save('{}/model/{}/{}/confusion_matrix_{}_{}_{}_use_cbam_{}'.format(workspace, expt_name, getSampleRateString(
        sample_rate), feature_type, str(perm[0])+str(perm[1])+str(perm[2]), model_arch, use_cbam), confusion_matrix)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Feature type')
    parser.add_argument('-en', '--expt_name', type=str, required=True)
    parser.add_argument('-w', '--workspace', type=str, default=workspace)
    parser.add_argument('-f', '--feature_type', type=str, default=feature_type)
    parser.add_argument('-n', '--num_frames', type=int, default=num_frames)
    parser.add_argument('-ma', '--model_arch', type=str, default='mobilenetv2')
    parser.add_argument('-msuffix','--model_suffix',type=str,default='')
    parser.add_argument('-cbam', '--use_cbam', action='store_true')
    parser.add_argument('-p', '--permutation', type=int,
                        nargs='+', default=permutation)
    parser.add_argument('-sr', '--sample_rate', type=int,
                        help="Specifies sample rates of the spectrogram.", default=sample_rate)
    parser.add_argument('-hop', '--hop_length', type=int,
                        help="Specifies hop length of the spectrogram.", default=hop_length)
    parser.add_argument('-kn','--knotes',type=int,default=1)
    parser.add_argument('-pnotes','--process_notes',type=bool,default=False)
    parser.add_argument('-gAll','--gaussian_all',type=bool,default=False)
    parser.add_argument('-val','--use_val',type=bool,default=False)
    parser.add_argument('-dp','--dropout',type=float,default=0)
    parser.add_argument('-dbmlp','--double_mlp',type=bool,default=False)
    parser.add_argument('-lpool','--layer_pooled',type=bool,default=[True,True,True,True],nargs='+')
    parser.add_argument('-lin','--linear_merge',type=int,default=1)
    parser.add_argument('-ign','--ignored_labels',type=str,default=[],nargs='+')
    args = parser.parse_args()
    sample_rate = args.sample_rate
    hop_length = args.hop_length

    run(args.workspace, args.feature_type, args.num_frames,
        args.permutation, args.model_arch, args.use_cbam, args.expt_name,args)
