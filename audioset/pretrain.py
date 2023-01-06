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
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import argparse
from utils import AudioDataset, Task5Model, configureTorchDevice, getSampleRateString, BalancedBatchSampler
from augmentation.SpecTransforms import TimeMask, FrequencyMask, RandomCycle
from torchsummary import summary
from config import feature_type, num_frames, seed, permutation, batch_size, num_workers, num_classes, learning_rate, amsgrad, patience, verbose, epochs, workspace, sample_rate, early_stopping, grad_acc_steps, model_arch, pann_cnn10_encoder_ckpt_path, pann_cnn14_encoder_ckpt_path, resume_training, n_mels, use_cbam, use_resampled_data, hop_length
import wandb
import sklearn
from glob import glob

__author__ = "Andrew, Yan Zhen, Anushka and Soham"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen", "Tanmay Khandelwal"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Soham Tiwari"
__email__ = "soham.tiwari800@gmail.com"
__status__ = "Development"

def run(args):
    wandb.init(project="st-project-sec-pretrain")
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
    num_patchout = args.num_patchout
    patchout_size = args.patchout_size
    print(f'Using cbam: {use_cbam}')
    print(f'Using pna: {use_pna}')
    print(f'Using mixup: {args.use_mixup}')
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
    
    if use_resampled_data:

        file_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/data/{}/audio_{}/*.wav.npy'.format(workspace,
                               feature_type, getSampleRateString(sample_rate))))]
        # train_list, val_list = sklearn.model_selection.train_test_split(file_list, train_size=0.8, random_state = seed)

        # file_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/data/{}/audio_{}/*.wav.npy'.format(workspace,
        #                        feature_type, getSampleRateString(sample_rate))))]
        # file_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/*.wav.npy'.format(workspace,
        #                 feature_type, getSampleRateString(sample_rate))))]
        # train_list, val_list = sklearn.model_selection.train_test_split(file_list, train_size=0.8, random_state = seed)
        train_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/train_logmel/{}/*.wav.npy'.format(workspace,f'sr={sample_rate}_hop={hop_length}',
                                                                                                        feature_type, getSampleRateString(sample_rate))))]    
        val_list = [os.path.basename(p)[:-8] for p in np.unique(glob('{}/val_logmel/{}/*.wav.npy'.format(workspace,f'sr={sample_rate}_hop={hop_length}',
                                                                                                feature_type, getSampleRateString(sample_rate))))]

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

    spec_transforms = transforms.Compose([
        TimeMask(),
        FrequencyMask(),
        RandomCycle()
    ])

    # albumentations_transform = Compose([
    #     ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.5),
    #     GridDistortion(),
    #     ToTensor()
    # ])

    # Create the datasets and the dataloaders


    train_dataset = AudioDataset(workspace, train_df, feature_type=feature_type,
                                 perm=perm,
                                 resize=num_frames,
                                #  image_transform=albumentations_transform,
                                #  spec_transform=spec_transforms
                                )

    valid_dataset = AudioDataset(
        workspace, valid_df, feature_type=feature_type, perm=perm, resize=num_frames)

    train_dataset = AudioDataset(
        workspace, train_df, feature_type=feature_type, perm=perm, resize=num_frames, usage='train', sample_rate=sample_rate, hop_length=hop_length)
    valid_dataset = AudioDataset(
        workspace, valid_df, feature_type=feature_type, perm=perm, resize=num_frames, usage='val', sample_rate=sample_rate, hop_length=hop_length)


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
    elif model_arch == 'pann_cnn10':
        model = Task5Model(num_classes, model_arch, pann_cnn10_encoder_ckpt_path=pann_cnn10_encoder_ckpt_path, use_cbam=use_cbam, use_pna = use_pna).to(device)
    elif model_arch == 'pann_cnn14':
        model = Task5Model(num_classes, model_arch, pann_cnn14_encoder_ckpt_path=pann_cnn14_encoder_ckpt_path, use_cbam=use_cbam, use_pna = use_pna).to(device)
    elif model_arch == 'passt':
        model = Task5Model(num_classes, model_arch).to(device)
    elif model_arch == "upasst":
        model = Task5Model(num_classes, model_arch).to(device)
    elif model_arch == "utoken":
        model = Task5Model(num_classes, model_arch).to(device)
    print(f'Using {model_arch} model.')
#     summary(model, (1, n_mels, num_frames))
    wandb.watch(model, log_freq=100)
    folderpath = '{}/model/{}/{}'.format(workspace, expt_name,
                                      getSampleRateString(sample_rate))
    os.makedirs(folderpath, exist_ok=True)
    model_path = '{}/model_{}_{}_{}_use_cbam_{}'.format(folderpath,
                                            feature_type, str(perm[0])+str(perm[1])+str(perm[2]), model_arch, use_cbam)

    # Define optimizer, scheduler and loss criteria
    if model_arch == 'upasst':
        optimizer = optim.Adam(
            model.passt.parameters(), lr=learning_rate, amsgrad=amsgrad)
        optimizer2 = optim.Adam(
            model.unet.parameters(), lr=learning_rate*10, amsgrad=amsgrad)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=epochs, eta_min=1e-7)
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, amsgrad=amsgrad)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=patience, verbose=verbose)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss()

    train_loss_hist = []
    valid_loss_hist = []
    lowest_val_loss = np.inf
    epochs_without_new_lowest = 0
    higest_val_acc = -np.inf
    if resume_training and os.path.exists(model_path):
        print(f'resume_training = {resume_training} using path {model_path}')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        starting_epoch = checkpoint['epoch']

    optimizer.zero_grad()
    for epoch in range(starting_epoch, starting_epoch+epochs):
        print('Epoch: ', epoch, 'LR: ', optimizer.param_groups[0]['lr'])

        this_epoch_train_loss = 0
        this_epoch_train_acc = 0
        batch = 0
        for sample in tqdm(train_loader):
            batch += 1
            inputs = sample['data'].to(device)
            label = sample['labels'].to(device)

            with torch.set_grad_enabled(True):
                model = model.train()
                # print(inputs.shape)
                # print(inputs)
                # outputs = model(inputs)
                x = inputs
                b, c, h, w = x.shape
                quantized = model(x)
                masked_x = x.clone().to(device)
                for i in range(b):
                    masked_row,masked_col = torch.randint(0, h//patchout_size[0], (num_patchout,), requires_grad=False),torch.randint(0, w//patchout_size[1], (num_patchout,),requires_grad=False)
                    for _r,_c in zip(masked_row,masked_col):
                        masked_x[i,_r*patchout_size[0]:(_r+1)*patchout_size[0],_c*patchout_size[1]:(_c+1)*patchout_size[1]] = 0
                masked_quantized = model(masked_x)
                loss = torch.zeros(1,).to(device)
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                for i in range(b):
                    loss += -torch.log(torch.exp(torch.mean(cos(quantized[i],masked_quantized[i])))/torch.sum(
                                torch.exp(torch.cat(tuple(torch.mean(cos(quantized[j],masked_quantized[i])).unsqueeze(0) for j in range(b) if i==j or label[i]!=label[j])))))
                    # loss += torch.log(torch.sum(torch.exp(torch.cat(tuple(torch.mean(cos(quantized[j],quantized[i])) for j in range(b) if i!=j)))))
                loss/=b
                # loss = criterion(outputs, label)
                loss.backward()
                if batch % grad_acc_steps == 0 or batch % len(train_loader) == 0:
                    if model_arch == 'upasst':
                        optimizer2.step()
                        optimizer2.zero_grad()
                    optimizer.step()
                    optimizer.zero_grad()
                if batch % 100 == 0:
                    print(f'train loss batch: {batch}, loss: {loss.detach().cpu().numpy()}')
                this_epoch_train_loss += loss.detach().cpu().numpy()
                # this_epoch_train_acc += ((outputs.argmax(dim=1) == label)*1.0).mean().detach().cpu().numpy()
        
        # this_epoch_train_acc /= batch
        this_epoch_train_loss /= len(train_df)
        wandb.log({"train":{
            "loss": this_epoch_train_loss,
            # "precision": this_epoch_train_acc
        }})
        batch = 0
        this_epoch_valid_loss = 0
        # this_epoch_valid_acc = 0
        for sample in tqdm(val_loader):
            batch += 1
            inputs = sample['data'].to(device)
            labels = sample['labels'].to(device)
            with torch.set_grad_enabled(False):
                model = model.eval()
                x = inputs
                b, c, h, w = x.shape
                quantized = model(x)
                masked_x = x.clone().to(device)
                for i in range(b):
                    masked_row,masked_col = torch.randint(0, h//patchout_size[0], (num_patchout,), requires_grad=False),torch.randint(0, w//patchout_size[1], (num_patchout,),requires_grad=False)
                    for _r,_c in zip(masked_row,masked_col):
                        masked_x[i,_r*patchout_size[0]:(_r+1)*patchout_size[0],_c*patchout_size[1]:(_c+1)*patchout_size[1]] = 0
                masked_quantized = model(masked_x)
                loss = torch.zeros(1,).to(device)
                cos = nn.CosineSimilarity(dim=1, eps=1e-6)
                for i in range(b):
                    loss += -torch.log(torch.exp(torch.mean(cos(quantized[i],masked_quantized[i])))/torch.sum(torch.exp(torch.cat(
                                            tuple(torch.mean(cos(quantized[j],masked_quantized[i])).unsqueeze(0) for j in range(b) if i==j or label[i]!=label[j])))))
                    # loss += torch.log(torch.sum(torch.exp(torch.cat(tuple(torch.mean(cos(quantized[j],quantized[i])) for j in range(b) if i!=j)))))
                loss/=b
                # outputs = model(inputs)
                # loss = criterion(outputs, labels)
                this_epoch_valid_loss += loss.detach().cpu().numpy()
                # this_epoch_valid_acc += ((outputs.argmax(dim=1) == labels)*1.0).mean().detach().cpu().numpy()


        # this_epoch_valid_acc /= batch
        this_epoch_valid_loss /= len(valid_df)

        wandb.log({"validation":{
            "loss": this_epoch_valid_loss,
            # "precision": this_epoch_valid_acc
        }})
        # print(f"train_loss = {this_epoch_train_loss}, val_loss={this_epoch_valid_loss}, precision={this_epoch_valid_acc}")
        print(f"train_loss = {this_epoch_train_loss}, val_loss={this_epoch_valid_loss}")
        train_loss_hist.append(this_epoch_train_loss)
        valid_loss_hist.append(this_epoch_valid_loss)

        # if this_epoch_valid_loss < lowest_val_loss:
        #     lowest_val_loss = this_epoch_valid_loss
        # if this_epoch_valid_acc>higest_val_acc:
        #     higest_val_acc = this_epoch_valid_acc
        if True:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, model_path)
            print(f'Saving model state at epoch: {epoch}.')
            epochs_without_new_lowest = 0
        else:
            epochs_without_new_lowest += 1

        if epochs_without_new_lowest >= early_stopping:
            break

        if model_arch == 'upasst':
            scheduler2.step(i)
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
    parser.add_argument('-num_p','--num_patchout', type=int, default=0)
    parser.add_argument('-psize','--patchout_size', type=tuple, default=(8,8))
    parser.add_argument('-batch','--batch_size', type=int, default=batch_size)
    parser.add_argument('-epoch','--epochs', type=int, default=epochs)
    args = parser.parse_args()


    sample_rate = args.sample_rate
    hop_length = args.hop_length
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs
    logging.getLogger().setLevel(logging.ERROR)

    run(args)
