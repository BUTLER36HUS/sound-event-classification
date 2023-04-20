#!/usr/bin/env python
import pandas as pd
import numpy as np
from zmq import device
# from albumentations import Compose, ShiftScaleRotate, GridDistortion
# from albumentations.pytorch import ToTensor
import logging
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import argparse
from utils import AudioDataset, Task5Model, configureTorchDevice, getSampleRateString, BalancedBatchSampler, AudioWavDataset
from augmentation.SpecTransforms import TimeMask, FrequencyMask, RandomCycle
from torchsummary import summary
from config import feature_type, num_frames, seed, permutation, batch_size, num_workers,\
      num_classes, class_mapping, learning_rate, amsgrad, patience, verbose, epochs, workspace, sample_rate, \
        early_stopping, grad_acc_steps, model_arch, pann_cnn10_encoder_ckpt_path, pann_cnn14_encoder_ckpt_path, \
            resume_training, n_mels, use_cbam, use_resampled_data, hop_length,use_raw_wav
import wandb
import sklearn
from glob import glob


from sklearn.metrics import f1_score

__author__ = "Andrew, Yan Zhen, Anushka and Soham"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Soham Tiwari"
__email__ = "soham.tiwari800@gmail.com"
__status__ = "Development"

def run(args):
    wandb.init(project="st-project-sec",name=args.expt_name)
    wandb.config.update(args)
    expt_name = args.expt_name
    workspace = args.workspace
    feature_type = args.feature_type
    num_frames = args.num_frames
    perm = args.permutation
    seed = args.seed
    resume_training = args.resume_training
    grad_acc_steps = args.grad_acc_steps
    model_arch = args.model_arch
    use_cbam = args.use_cbam
    use_pna = args.use_pna
    sample_rate  = args.sample_rate
    hop_length = args.hop_length
    print(f'Using cbam: {use_cbam}')
    print(f'Using pna: {use_pna}')
    print(f'Using mixup: {args.use_mixup}')
    print(args)
    if model_arch == 'pann_cnn10':
        pann_cnn10_encoder_ckpt_path = args.pann_cnn10_encoder_ckpt_path
    elif model_arch == 'pann_cnn14':
        pann_cnn14_encoder_ckpt_path = args.pann_cnn14_encoder_ckpt_path
    balanced_sampler = args.balanced_sampler
    balanced_sampler_val = args.balanced_sampler_val

    starting_epoch = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.makedirs('{}/model'.format(workspace), exist_ok=True)
    
    removed_train = set()
    removed_valid = set()




    if use_resampled_data:
        ignored_labels = args.ignored_labels
        for label in ignored_labels:
            if label in class_mapping:
                del class_mapping[label]
        num_classes = max(class_mapping.values())+1
        file_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/data/{}/audio_{}/*.wav.npy'.format(workspace,
                               feature_type, getSampleRateString(sample_rate))))]
        train_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/train_logmel/{}/*.wav.npy'.format(workspace,f'sr={sample_rate}_hop={hop_length}',
                                                                                                        feature_type, getSampleRateString(sample_rate)))) if os.path.basename(p).split('-')[0] not in ignored_labels]    
        val_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/val_logmel/{}/*.wav.npy'.format(workspace,f'sr={sample_rate}_hop={hop_length}',
                                                                                                feature_type, getSampleRateString(sample_rate)))) if os.path.basename(p).split('-')[0] not in ignored_labels ]

        train_df = pd.DataFrame(train_list)
        valid_df = pd.DataFrame(val_list)
    elif use_raw_wav:
        ignored_labels = args.ignored_labels
        for label in ignored_labels:
            if label in class_mapping:
                del class_mapping[label]
        num_classes = max(class_mapping.values())+1
        train_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/train_embed/*.wav.npy'.format(workspace,))) if os.path.basename(p).split('-')[0] not in ignored_labels]    
        val_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/val_embed/*.wav.npy'.format(workspace,))) if os.path.basename(p).split('-')[0] not in ignored_labels ]

        train_df = pd.DataFrame(train_list)
        valid_df = pd.DataFrame(val_list)
    else:
        folds = []
        for i in range(5):
            folds.append(pd.read_csv(
                '{}/split/fold_{}_c.txt'.format(workspace, i), delimiter=" ", header=None))

        train_df = pd.concat([folds[perm[0]], folds[perm[1]], folds[perm[2]]])
        valid_df = folds[perm[3]]
    
    # test_df = folds[perm[4]]

    spec_transforms = None
    if not args.no_transform:
        spec_transforms = transforms.Compose([
        TimeMask(),
        FrequencyMask(),
        RandomCycle(),
        ])
    if args.gaussian_all:
        spec_transforms = transforms.Compose([
            transforms.GaussianBlur((5,5)),
        ])

    # albumentations_transform = Compose([
    #     ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.5),
    #     GridDistortion(),
    #     ToTensor()
    # ])

    # Create the datasets and the dataloaders


    # train_dataset = AudioDataset(workspace, train_df, feature_type=feature_type,
    #                              perm=perm,
    #                              resize=num_frames,
    #                             #  image_transform=albumentations_transform,
    #                              spec_transform=spec_transforms)

    # valid_dataset = AudioDataset(
    #     workspace, valid_df, feature_type=feature_type, perm=perm, resize=num_frames, spec_transform = spec_transforms if args.gaussian_all else None)

    if use_raw_wav:
        train_dataset = AudioWavDataset(workspace,train_df,usage='train')
        valid_dataset = AudioWavDataset(workspace,valid_df,usage='val')
    else:
        train_dataset = AudioDataset(
            workspace, train_df, feature_type=feature_type, perm=perm, resize=num_frames, usage='train', 
                    sample_rate=sample_rate, hop_length=hop_length, spec_transform=spec_transforms)
        valid_dataset = AudioDataset(
            workspace, valid_df, feature_type=feature_type, perm=perm, resize=num_frames, usage='val', 
                    sample_rate=sample_rate, hop_length=hop_length, spec_transform=spec_transforms if args.gaussian_all else None)


    print(f'Using balanced_sampler = {balanced_sampler}')
    if balanced_sampler:
        train_loader = DataLoader(train_dataset, batch_size, sampler=BalancedBatchSampler(train_df), num_workers=num_workers, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    print(f'Using balanced_sampler_val = {balanced_sampler_val}')
    
    # val_loader = DataLoader(valid_dataset, batch_size,
    #                         shuffle=False, num_workers=num_workers)
    if balanced_sampler_val:
        val_loader = DataLoader(valid_dataset, batch_size, sampler=BalancedBatchSampler(valid_df,useMixup=args.mixup), num_workers=num_workers, shuffle=False)
    else:
        val_loader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers)

    # Define the device to be used
    device = configureTorchDevice()
    # Instantiate the model
    if model_arch == 'mobilenetv2' or model_arch == 'mobilenetv3':
        model = Task5Model(num_classes, model_arch, use_cbam=use_cbam).to(device)
    elif model_arch == 'multimobilenetv3' or model_arch=="multimobilenetv2":
        model = Task5Model(num_classes, model_arch,dropout=args.dropout).to(device)
    elif model_arch == 'pann_cnn10':
        model = Task5Model(num_classes, model_arch, pann_cnn10_encoder_ckpt_path=pann_cnn10_encoder_ckpt_path, use_cbam=use_cbam, use_pna = use_pna).to(device)
    elif model_arch == 'pann_cnn14':
        model = Task5Model(num_classes, model_arch, pann_cnn14_encoder_ckpt_path=pann_cnn14_encoder_ckpt_path, use_cbam=use_cbam, use_pna = use_pna).to(device)
    elif model_arch.startswith("noted"):
        model = Task5Model(num_classes, model_arch,dropout=args.dropout).to(device)
    elif model_arch.startswith("resnet"):
        model = Task5Model(num_classes, model_arch,dropout=args.dropout).to(device)
    elif model_arch.startswith("multipool"):
        model = Task5Model(num_classes, model_arch,dropout=args.dropout,layer_pooled=args.layer_pooled,linear_merge=args.linear_merge).to(device)
    elif model_arch == "CLAP":
        model = Task5Model(num_classes, model_arch,dropout=args.dropout).to(device)
        # model.CLAP.to(device)

    print(f'Using {model_arch} model.')
#     summary(model, (1, n_mels, num_frames))
    wandb.watch(model, log_freq=100)
    folderpath = '{}/model/{}/{}'.format(workspace, expt_name,
                                      getSampleRateString(sample_rate))
    os.makedirs(folderpath, exist_ok=True)
    model_path = '{}/model_{}_{}_{}_use_cbam_{}'.format(folderpath,
                                            feature_type, str(perm[0])+str(perm[1])+str(perm[2]), model_arch, use_cbam)

    # Define optimizer, scheduler and loss criteria
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, amsgrad=amsgrad, weight_decay=args.weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, patience=patience, verbose=verbose)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss()

    train_loss_hist = []
    valid_loss_hist = []
    lowest_val_loss = np.inf
    epochs_without_new_lowest = 0
    higest_val_f1 = -1
    if resume_training and os.path.exists(model_path):
        print(f'resume_training = {resume_training} using path {model_path}')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']

    
    TENS_LOG = np.log(10)
    optimizer.zero_grad()
    for i in range(starting_epoch, starting_epoch+epochs):
        print('Epoch: ', i, 'LR: ', optimizer.param_groups[0]['lr'])

        wandb.log({"optimizer":{
            "lr": optimizer.param_groups[0]['lr'],
        }})
        this_epoch_train_loss = 0
        this_epoch_train_acc = 0
        # this_epoch_train_f1 = 0
        batch = 0
        for sample in tqdm(train_loader,mininterval=len(train_dataset)//100):
            batch += 1
            inputs = sample['data'].to(device)
            label = sample['labels'].to(device)

            with torch.set_grad_enabled(True):
                model = model.train()
                outputs = model(inputs)
                loss = criterion(outputs, label)
                loss.backward()
                if batch % grad_acc_steps == 0 or batch % len(train_loader) == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                this_epoch_train_loss += loss.detach().cpu().numpy()
                this_epoch_train_acc += ((outputs.argmax(dim=1) == label)*1.0).mean().detach().cpu().numpy()
                # this_epoch_train_f1 += f1_score(label.detach().cpu().numpy(), outputs.argmax(dim=1).detach().cpu().numpy(), average='macro')
        
        this_epoch_train_acc /= batch
        # this_epoch_train_f1 /= batch


        batch = 0
        this_epoch_valid_loss = 0
        this_epoch_valid_acc = 0
        this_epoch_valid_f1 = 0
        y_pred = []
        y_true = []
        for sample in tqdm(val_loader):
            batch += 1
            inputs = sample['data'].to(device)
            labels = sample['labels'].to(device)
            y_true += labels.detach().cpu().numpy().tolist()
            with torch.set_grad_enabled(False):
                model = model.eval()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                this_epoch_valid_loss += loss.detach().cpu().numpy()
                this_epoch_valid_acc += ((outputs.argmax(dim=1) == labels)*1.0).mean().detach().cpu().numpy()
                y_pred += outputs.argmax(dim=1).detach().cpu().numpy().tolist()

        this_epoch_train_loss /= len(train_df)
        this_epoch_valid_acc /= batch
        this_epoch_valid_f1 = f1_score(y_true, y_pred, average='macro')
        this_epoch_valid_f1 /= batch
        this_epoch_valid_loss /= len(valid_df)
        wandb.log({"train":{
            "loss": this_epoch_train_loss,
            "precision": this_epoch_train_acc,
            # "f1": this_epoch_train_f1
        }})
        wandb.log({"validation":{
            "loss": this_epoch_valid_loss,
            "precision": this_epoch_valid_acc,
            "f1": this_epoch_valid_f1
        }})
        print(
            f"train_loss = {this_epoch_train_loss}, val_loss={this_epoch_valid_loss}, precision={this_epoch_valid_acc}")
        train_loss_hist.append(this_epoch_train_loss)
        valid_loss_hist.append(this_epoch_valid_loss)

        if this_epoch_valid_loss < lowest_val_loss or this_epoch_valid_f1>higest_val_f1:
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, model_path+("_loss" if this_epoch_valid_f1<higest_val_f1 else "" ))
            print(f'Saving model state at epoch: {i}.')
            lowest_val_loss = this_epoch_valid_loss
            epochs_without_new_lowest = 0
            higest_val_f1 = this_epoch_valid_f1
            print(f'Saving model state at epoch: {i}.')
            epochs_without_new_lowest = 0
        else:
            epochs_without_new_lowest += 1

        if args.early_stopping and  epochs_without_new_lowest >= early_stopping:
            break

        if epochs%args.scheduler_duration == 0:
            scheduler.step(this_epoch_valid_loss)
            # scheduler.step(epoch=i)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Feature type')
    parser.add_argument('-en', '--expt_name', type=str, required=True)
    parser.add_argument('-w', '--workspace', type=str, default=workspace)
    parser.add_argument('-f', '--feature_type', type=str, default=feature_type)
    parser.add_argument('-ma', '--model_arch', type=str, default=model_arch)
    parser.add_argument('-kwargs', '--model_kwargs', type=str, default={}, nargs='+')
    parser.add_argument('-cp10', '--pann_cnn10_encoder_ckpt_path',
                        type=str, default=pann_cnn10_encoder_ckpt_path)
    parser.add_argument('-cp14', '--pann_cnn14_encoder_ckpt_path',
                        type=str, default=pann_cnn14_encoder_ckpt_path)
    parser.add_argument('-n', '--num_frames', type=int, default=num_frames)
    parser.add_argument('-p', '--permutation', type=int,
                        nargs='+', default=permutation) 
    parser.add_argument('-s', '--seed', type=int, default=seed)
    parser.add_argument('-rt', '--resume_training', action='store_true')
    parser.add_argument('-bs', '--balanced_sampler', type=bool, default=False)
    parser.add_argument('-bsval', '--balanced_sampler_val', type=bool, default=False)
    parser.add_argument('-mixup', '--use_mixup', type=bool, default=False)
    parser.add_argument('-cbam', '--use_cbam', action='store_true')
    parser.add_argument('-pna', '--use_pna', action='store_true')
    parser.add_argument('-ga', '--grad_acc_steps',
                        type=int, default=grad_acc_steps)
    parser.add_argument('-sr', '--sample_rate', type=int,
                        help="Specifies sample rates of the spectrogram.", default=sample_rate)
    parser.add_argument('-hop', '--hop_length', type=int,
                        help="Specifies hop length of the spectrogram.", default=hop_length)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-batch','--batch_size', type=int, default=batch_size)
    parser.add_argument('-epoch','--epochs', type=int, default=epochs)
    parser.add_argument('-kn','--knotes',type=int,default=1)
    parser.add_argument('-l2','--weight_decay',type=float,default=0)
    parser.add_argument('-dp','--dropout',type=float,default=0)
    parser.add_argument('-pnotes','--process_notes',type=bool,default=False)
    parser.add_argument('-gAll','--gaussian_all',type=bool,default=False)
    parser.add_argument('-notrans','--no_transform',type=bool,default=False)
    parser.add_argument('-early','--early_stopping',type=bool,default=True)
    parser.add_argument('-sch','--scheduler_duration',type=int,default=1)
    # parser.add_argument('-dbmlp','--double_mlp',type=bool,default=False)
    parser.add_argument('-lpool','--layer_pooled',type=bool,default=[True],nargs='+')
    parser.add_argument('-lin','--linear_merge',type=int,default=1)
    # parser.add_argument('-lw','--loss_weight',type=float,default=[1.0],nargs='+')
    parser.add_argument('-lw','--loss_weights',type=float,default=[1.0],nargs='+')
    parser.add_argument('-ign','--ignored_labels',type=str,default=[],nargs='+')
    args = parser.parse_args()


    sample_rate = args.sample_rate
    hop_length = args.hop_length
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    logging.getLogger().setLevel(logging.ERROR)

    run(args)
