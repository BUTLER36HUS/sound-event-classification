from typing import Iterator
from matplotlib.style import use
import pandas as pd
import scipy
from config import feature_type, permutation, sample_rate, num_frames, use_cbam, cbam_kernel_size, cbam_reduction_factor, use_median_filter, use_pna, model_archs, class_mapping,sample_rate,hop_length
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision
from augmentation.SpecTransforms import ResizeSpectrogram
from augmentation.RandomErasing import RandomErasing
from attention.CBAM import CBAMBlock
from pann_encoder import Cnn10, Cnn14
from ParNetAttention import ParNetAttention
from dynamic_convolutions import Dynamic_conv2d
import os

from model.passt.passt import PaSST
from model.unet.unet_model import UNet
from model.utoken import UToken

__author__ = "Andrew Koh Jin Jie, Yan Zhen"
__credits__ = ["Prof Chng Eng Siong", "Yan Zhen",
               "Tanmay Khandelwal", "Anushka Jain"]
__license__ = "GPL"
__version__ = "0.0.0"
__maintainer__ = "Soham Tiwari"
__email__ = "soham.tiwari800@gmail.com"
__status__ = "Development"


random_erasing = RandomErasing()


def getFileNameFromDf(df: pd.DataFrame, idx: int) -> str:
    """Returns filename for the audio file at index idx of df

    Args:
        df (pd.Dataframe): df of audio files
        idx (int): index of audio file in df

    Returns:
        str: file name of audio file at index 'idx' in df.
    """
    curr = df.iloc[idx, :]
    file_name = curr[0]
    return file_name


def getLabelFromFilename(file_name: str) -> int:
    """Extracts the label from the filename

    Args:
        file_name (str): audio file name

    Returns:
        int: integer label for the audio file name
    """
    label = class_mapping[file_name.split('-')[0]]
    return label


class AudioDataset(Dataset):



    def __init__(self, workspace, df, feature_type=feature_type, perm=permutation, spec_transform=None, image_transform=None, resize=num_frames, sample_rate=sample_rate, hop_length=hop_length,
                    usage:str='train'):


        self.workspace = workspace
        self.df = df
        self.filenames = df[0].unique()
        self.length = len(self.filenames)
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.spec_transform = spec_transform
        self.image_transform = image_transform
        self.resize = ResizeSpectrogram(frames=resize)
        self.pil = transforms.ToPILImage()


        # self.channel_means = np.load('{}/data/statistics/{}/channel_means_{}_{}.npy'.format(
        #     workspace, getSampleRateString(sample_rate), feature_type, str(perm[0])+str(perm[1])+str(perm[2])))
        # self.channel_stds = np.load('{}/data/statistics/{}/channel_stds_{}_{}.npy'.format(
        #     workspace, getSampleRateString(sample_rate), feature_type, str(perm[0])+str(perm[1])+str(perm[2])))

        self.channel_means = np.load('{}/{}_logmel/{}/data/statistics/{}/channel_means_{}_{}.npy'.format(
            workspace, usage, f'sr={sample_rate}_hop={hop_length}',getSampleRateString(sample_rate), feature_type, str(perm[0])+str(perm[1])+str(perm[2])))
        self.channel_stds = np.load('{}/{}_logmel/{}/data/statistics/{}/channel_stds_{}_{}.npy'.format(
            workspace, usage, f'sr={sample_rate}_hop={hop_length}',getSampleRateString(sample_rate), feature_type, str(perm[0])+str(perm[1])+str(perm[2])))


        self.channel_means = self.channel_means.reshape(1, -1, 1)
        self.channel_stds = self.channel_stds.reshape(1, -1, 1)
        self.usage = usage

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        if idx<self.length:
            file_name = getFileNameFromDf(self.df, idx)
            labels = getLabelFromFilename(file_name)
            # sample = np.load(
            #     f"{self.workspace}/data/{self.feature_type}/audio_{getSampleRateString(self.sample_rate)}/{file_name}.wav.npy")

            sample = np.load(
                f"{self.workspace}/{self.usage}_logmel/sr={self.sample_rate}_hop={self.hop_length}/{file_name}.wav.npy")
        else:
            temp = idx
            file_names = []
            while temp>0:
                file_names.append(getFileNameFromDf(self.df,temp%self.length))
                temp = temp//self.length
            labels = getLabelFromFilename(file_names[0])
            sample = np.array([np.load(
                                    f"{self.workspace}/{self.usage}_logmel/sr={self.sample_rate}_hop={self.hop_length}/{file_name}.wav.npy")
                                            for file_name in file_names])
            sample = np.mean(sample,axis=0)



        if self.resize:
            sample = self.resize(sample)

        sample = (sample-self.channel_means)/self.channel_stds
        sample = torch.Tensor(sample)

        if self.spec_transform:
            sample = self.spec_transform(sample)

        if self.image_transform:
            # min-max transformation
            this_min = sample.min()
            this_max = sample.max()
            sample = (sample - this_min) / (this_max - this_min)

            # randomly cycle the file
            i = np.random.randint(sample.shape[1])
            sample = torch.cat([
                sample[:, i:, :],
                sample[:, :i, :]],
                dim=1)

            # apply albumentations transforms
            sample = np.array(self.pil(sample))
            sample = self.image_transform(image=sample)
            sample = sample['image']
            sample = sample[None, :, :].permute(0, 2, 1)

            # apply random erasing
            sample = random_erasing(sample.clone().detach())

            # revert min-max transformation
            sample = (sample * (this_max - this_min)) + this_min

        if len(sample.shape) < 3:
            sample = torch.unsqueeze(sample, 0)

        labels = torch.LongTensor([labels]).squeeze()

        data = {}
        data['data'], data['labels'], data['file_name'] = sample, labels, file_name
        return data


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset_df, maxdb=120.0, mindb=-50, useMixup=False):
        self.df = dataset_df
        self.filenames = self.df[0].unique()
        self.length = len(self.filenames)
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, self.length):
            label = self._get_label(idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            if len(self.dataset[label]) > self.balanced_max:
                self.balanced_max = len(self.dataset[label])


        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            diff = self.balanced_max - len(self.dataset[label])
            if diff > 0:
                if useMixup:
                    raw_len = len(self.dataset[label])
                    mix_num = 2
                    while diff > 0 and raw_len>mix_num:
                        self.dataset[label].extend(self.naive_mixup(label,mix_num=mix_num,raw_length=raw_len)[:min(diff,raw_len**mix_num)])
                        mix_num+=1
                        diff = self.balanced_max - len(self.dataset[label])
                if diff>0: # still need to oversample (the number of samples is less than mix_num)aa
                    self.dataset[label].extend(
                        np.random.choice(self.dataset[label], size=diff))
                    
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1]*len(self.keys)

    def __iter__(self) -> Iterator[int]:
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1]*len(self.keys)

    def _get_label(self, idx):
        file_name = getFileNameFromDf(self.df, idx)
        label = getLabelFromFilename(file_name)
        return label

    def __len__(self):
        return self.balanced_max*len(self.keys)

    def naive_mixup(self,label, mix_num, raw_length):
        raws = self.dataset[label][:raw_length]
        gradients = np.choice(raws, size=mix_num*(raw_length**mix_num)).reshape(mix_num, -1)
        idx_factors = (self.length**np.arange(mix_num)).reshape(mix_num, 1)
        idxs = np.sum(gradients*idx_factors, axis=0)+self.length
        return idxs



def apply_median(self, predictions):
    """To apply standard median filtering.

    Args:
        predictions (torch.Tensor): Predictions

    Returns:
        torch.Tensor: Predictons smoothed using median filter
    """
    device = predictions.device
    predictions = predictions.cpu().detach().numpy()
    for batch in range(predictions.shape[0]):
        predictions[batch, ...] = scipy.ndimage.filters.median_filter(
            predictions[batch, ...])

    return torch.from_numpy(predictions).float().to(device)


class Task5Model(nn.Module):

    def __init__(self, num_classes, model_arch: str = model_archs[0], pann_cnn10_encoder_ckpt_path: str = '', pann_cnn14_encoder_ckpt_path: str = '', use_cbam: bool = use_cbam, use_pna: bool = use_pna, use_median_filter: bool = use_median_filter,
                    dataset=None,dataset_sampler = None, dropout:float = 0.0, **kwargs):
        """Initialising model for Task 5 of DCASE

        Args:
            num_classes (int): Number of classes_
            model_arch (str, optional): Model architecture to be used. One of ['mobilenetv2', 'pann_cnn10', 'pann_cnn14']. Defaults to model_archs[0].
            pann_cnn10_encoder_ckpt_path (str, optional): File path for downloaded pretrained model checkpoint. Defaults to None.
            pann_cnn14_encoder_ckpt_path (str, optional): File path for downloaded pretrained model checkpoint. Defaults to None.

        Raises:
            Exception: Invalid model_arch paramater passed.
            Exception: Model checkpoint path does not exist/not found.
        """
        super().__init__()
        self.num_classes = num_classes

        if len(model_arch) > 0:
            if model_arch not in model_archs:
                raise Exception(
                    f'Invalid model_arch={model_arch} paramater. Must be one of {model_archs}')
            self.model_arch = model_arch

        self.use_cbam = use_cbam
        self.use_pna = use_pna
        self.use_median_filter = use_median_filter

        if self.model_arch == 'mobilenetv2':
            self.bw2col = nn.Sequential(
                Dynamic_conv2d(1, 10, 1, padding=0),
                Dynamic_conv2d(10, 3, 1, padding=0),
                nn.BatchNorm2d(3),
                CBAMBlock(
                    channel=3, reduction=cbam_reduction_factor, kernel_size=cbam_kernel_size) if self.use_cbam else nn.Identity()
            )
            
            self.mv2 = torchvision.models.mobilenet_v2(pretrained=True)

            if self.use_cbam:
                self.cbam = CBAMBlock(
                    channel=1280, reduction=cbam_reduction_factor, kernel_size=cbam_kernel_size)

            self.final = nn.Sequential(
                nn.Linear(1280, 512), nn.ReLU(), nn.BatchNorm1d(512),
                nn.Linear(512, num_classes))

        if self.model_arch == 'mobilenetv3':
            self.bw2col = nn.Sequential(
                Dynamic_conv2d(1, 10, 1, padding=0),
                Dynamic_conv2d(10, 3, 1, padding=0),
                nn.BatchNorm2d(3),
            )
            self.mv3 = torchvision.models.mobilenet_v3_large(pretrained=True)

            if self.use_cbam:
                self.cbam = CBAMBlock(
                    channel=960, reduction=cbam_reduction_factor, kernel_size=cbam_kernel_size)

            self.final = nn.Sequential(
                nn.Linear(960, 512), nn.ReLU(), nn.BatchNorm1d(512),
                nn.Linear(512, num_classes))
        
        elif  self.model_arch == "notedmobilenetv2":
            self.knotes = kwargs['knotes']
            self.process_notes = kwargs['process_notes']
            # input_channels = 1 + num_classes*self.knotes
            input_channels = num_classes*(1+self.knotes)
            notes = []
            for label in sorted(dataset_sampler.dataset.keys()):
                for idx_of_data_idx in np.random.randint(0, len(dataset_sampler.dataset[label]), size=self.knotes):
                    notes.append(dataset[dataset_sampler.dataset[label][idx_of_data_idx]]['data'])
            self.notes = nn.Parameter(torch.cat(notes), requires_grad=False)
                
            self.bw2col = nn.Sequential(
                # Dynamic_conv2d(input_channels, input_channels, 1, padding=0),
                # nn.BatchNorm2d(input_channels),
                Dynamic_conv2d(input_channels, input_channels*2, 1, padding=0),
                nn.Dropout(dropout),
                Dynamic_conv2d(input_channels*2, input_channels, 1, padding=0),
                Dynamic_conv2d(input_channels, 3, 1, padding=0),
                nn.BatchNorm2d(3),
                CBAMBlock(
                    channel=3, reduction=cbam_reduction_factor, kernel_size=cbam_kernel_size) if self.use_cbam else nn.Identity()
            )
            
            self.mv2 = torchvision.models.mobilenet_v2(pretrained=True)
            if self.process_notes==True:
                notes_num = num_classes*self.knotes
                self.note_processer = nn.Sequential(
                    nn.Conv2d(notes_num,2*notes_num,kernel_size=5,padding=2),
                    nn.GELU(),
                    nn.Conv2d(notes_num*2,notes_num*4,kernel_size=5,padding=2),
                    nn.GELU(),
                    nn.Conv2d(notes_num*4,notes_num,kernel_size=5,padding=2),
                )

            if self.use_cbam:
                self.cbam = CBAMBlock(
                    channel=1280, reduction=cbam_reduction_factor, kernel_size=cbam_kernel_size)

            self.final = nn.Sequential(
                nn.Linear(1280, 512), 
                nn.ReLU(), 
                nn.BatchNorm1d(512),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes))

        elif self.model_arch == "notedmobilenetv3":
            self.knotes = kwargs['knotes']
            self.process_notes = kwargs['process_notes']
            # input_channels = 1 + num_classes*self.knotes
            input_channels = num_classes*(1+self.knotes)
            notes = []
            for label in sorted(dataset_sampler.dataset.keys()):
                for idx_of_data_idx in np.random.randint(0, len(dataset_sampler.dataset[label]), size=self.knotes):
                    notes.append(dataset[dataset_sampler.dataset[label][idx_of_data_idx]]['data'])
            self.notes = nn.Parameter(torch.cat(notes), requires_grad=False)
                
            self.bw2col = nn.Sequential(
                # Dynamic_conv2d(input_channels, input_channels, 1, padding=0),
                # nn.BatchNorm2d(input_channels),
                Dynamic_conv2d(input_channels, input_channels*2, 1, padding=0),
                nn.Dropout(dropout),
                Dynamic_conv2d(input_channels*2, input_channels, 1, padding=0),
                Dynamic_conv2d(input_channels, 3, 1, padding=0),
                nn.BatchNorm2d(3),
            )
            self.mv3 = torchvision.models.mobilenet_v3_large(pretrained=True)

            if self.process_notes==True:
                notes_num = num_classes*self.knotes
                self.note_processer = nn.Sequential(
                    nn.Conv2d(notes_num,2*notes_num,kernel_size=5,padding=2),
                    nn.GELU(),
                    nn.Conv2d(notes_num*2,notes_num*4,kernel_size=5,padding=2),
                    nn.GELU(),
                    nn.Conv2d(notes_num*4,notes_num,kernel_size=5,padding=2),
                )

            if self.use_cbam:
                self.cbam = CBAMBlock(
                    channel=960, reduction=cbam_reduction_factor, kernel_size=cbam_kernel_size)

            self.final = nn.Sequential(
                nn.Linear(960, 512), 
                nn.ReLU(), 
                nn.BatchNorm1d(512),
                nn.Dropout(dropout),
                nn.Linear(512, num_classes))

        elif self.model_arch == 'pann_cnn10':
            if len(pann_cnn10_encoder_ckpt_path) > 0 and os.path.exists(pann_cnn10_encoder_ckpt_path) == False:
                raise Exception(
                    f"Model checkpoint path '{pann_cnn10_encoder_ckpt_path}' does not exist/not found.")
            self.pann_cnn10_encoder_ckpt_path = pann_cnn10_encoder_ckpt_path

            self.AveragePool = nn.AvgPool2d((1, 2), (1, 2))

            self.encoder = Cnn10()
            if self.pann_cnn10_encoder_ckpt_path != '':
                self.encoder.load_state_dict(torch.load(
                    self.pann_cnn10_encoder_ckpt_path)['model'], strict=False)
                print(
                    f'loaded pann_cnn14 pretrained encoder state from {self.pann_cnn10_encoder_ckpt_path}')

            if self.use_cbam:
                self.cbam = CBAMBlock(
                    channel=512, reduction=cbam_reduction_factor, kernel_size=cbam_kernel_size)

            if self.use_pna:
                self.pna = ParNetAttention(channel=512)
            
            self.pann_head = nn.Sequential(
                self.cbam if self.use_cbam else nn.Identity(),
                Dynamic_conv2d(512, 256, (1, 1)),
                Dynamic_conv2d(256, 128, (1, 1)),
            )
            # output shape of CNN10 [-1, 512, 39, 4]

            self.final = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
                nn.Linear(64, num_classes))

        elif self.model_arch == 'pann_cnn14':
            if len(pann_cnn14_encoder_ckpt_path) > 0 and os.path.exists(pann_cnn14_encoder_ckpt_path) == False:
                raise Exception(
                    f"Model checkpoint path '{pann_cnn14_encoder_ckpt_path}' does not exist/not found.")
            self.pann_cnn14_encoder_ckpt_path = pann_cnn14_encoder_ckpt_path

            self.AveragePool = nn.AvgPool2d((1, 2), (1, 2))

            self.encoder = Cnn14()
            if self.pann_cnn14_encoder_ckpt_path != '':
                self.encoder.load_state_dict(torch.load(
                    self.pann_cnn14_encoder_ckpt_path)['model'], strict=False)
                print(
                    f'loaded pann_cnn10 pretrained encoder state from {self.pann_cnn14_encoder_ckpt_path}')

            if self.use_cbam:
                self.cbam = CBAMBlock(
                    channel=2048, reduction=cbam_reduction_factor, kernel_size=cbam_kernel_size)

            if self.use_pna:
                self.pna = ParNetAttention(channel=2048)

            self.final = nn.Sequential(
                nn.Linear(2048, 512), nn.ReLU(), nn.BatchNorm1d(512),
                nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256),
                nn.Linear(256, num_classes))


        if model_arch=="passt" or model_arch=="upasst":
            self.passt = PaSST(
                num_classes=num_classes,
                # img_size = (128,1757),
                # u_patchout=2300,
                img_size= (128, 638),
                u_patchout=100,
                # s_patchout_f=4,
                stride=10,
                drop_rate=0.1,
                )
        
        if model_arch=="upasst":
            print("Using Unet")
            self.unet = UNet(1,1)

        
        if model_arch == "utoken":
            self.unet = UToken(1,1)
        
        if model_arch=="rfrnn":
            self.hidden_size = hidden_size = 128
            self.input_size = input_size = 128
            self.num_layers = num_layers = 4
            self.attn = nn.Sequential(
                nn.Linear(hidden_size,(hidden_size+input_size)*2),
                nn.Sigmoid(),
                nn.ReLU(),
                nn.Linear((hidden_size+input_size)*2,input_size),
                nn.Sigmoid()
            )
            self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=False)
            self.mlp = nn.Sequential(
                # nn.Linear(hidden_size*gru_layers, hidden_size*2),
                nn.Sigmoid(),
                nn.Linear(hidden_size, hidden_size*2),
                nn.ReLU(),
                nn.Linear(hidden_size*2, hidden_size*2),
                nn.ReLU(),
                nn.Linear(hidden_size*2, num_classes),
                #nn.Softmax(dim=1)
            )
            # self.rnns = ({
            #     'attn': nn.Sequential(
            #         nn.Linear(hidden_size,(hidden_size+input_size)*2),
            #         nn.Sigmoid(),
            #         nn.ReLU(),
            #         nn.Linear((hidden_size+input_size)*2,input_size),
            #         nn.Sigmoid()
            #     ),
            #     'rnn': nn.GRU(input_size=input_size, hidden_size=hidden_size, 
            #         num_layers=num_layers, batch_first=True, bidirectional=False),
            #     'mlp': nn.Sequential(
            #         # nn.Linear(hidden_size*gru_layers, hidden_size*2),
            #         nn.Linear(hidden_size, hidden_size*2),
            #         nn.Tanh(),
            #         nn.ReLU(),
            #         nn.Linear(hidden_size*2, hidden_size*2),
            #         nn.ReLU(),
            #         nn.Linear(hidden_size*2, 2),
            #         #nn.Softmax(dim=1)
            #     )} for _ in range(num_classes)
            # )



    def forward(self, x):
        if self.model_arch == 'mobilenetv2':
            x = self.bw2col(x)  # -> (batch_size, 3, n_mels, num_frames)
            x = self.mv2.features(x)

        elif self.model_arch == 'mobilenetv3':
            x = self.bw2col(x)  # -> (batch_size, 3, n_mels, num_frames)
            x = self.mv3.features(x)

        elif self.model_arch == 'pann_cnn10' or self.model_arch == 'pann_cnn14':
            x = x  # -> (batch_size, 1, n_mels, num_frames)
            x = x.permute(0, 1, 3, 2)  # -> (batch_size, 1, num_frames, n_mels)
            x = self.AveragePool(x)  # -> (batch_size, 1, num_frames, n_mels/2)
            # try to use a linear layer here.
            x = torch.squeeze(x, 1)  # -> (batch_size, num_frames, 64)
            x = self.encoder(x)
            x = self.pann_head(x)

        elif self.model_arch == "notedmobilenetv3" or self.model_arch=="notedmobilenetv2":
            repeated_notes = self.notes
            if self.process_notes==True:
                repeated_notes = self.note_processer(repeated_notes)
            repeated_notes = repeated_notes.unsqueeze(0).repeat(x.shape[0],1,1,1)
            # x = torch.cat([x,self.notes.unsqueeze(0).repeat(x.shape[0],1,1,1)],dim=1)
            x = torch.cat([torch.cat([x,repeated_notes[:,i*self.knotes:(i+1)*self.knotes]],dim=1) for i in range(self.num_classes)],dim=1)
            x = self.bw2col(x)
            x = self.mv3.features(x) if self.model_arch=="notedmobilenetv3" else self.mv2.features(x)

        
        elif self.model_arch=="passt":
            x = self.passt(x)[0]
            return x
        elif self.model_arch=="upasst":
            x = self.unet(x)
            x = self.passt(x)[0]
            return x
        elif self.model_arch=="utoken":
            return self.unet(x)
        elif self.model_arch == 'RFRNN':
            gru_output = torch.zeros((x.shape[0],1,self.hidden_size)).to(self.device)
            hidden = torch.zeros((self.num_layers,x.shape[0],self.hidden_size)).to(self.device)
            attn_norms = torch.zeros((1)).squeeze().to(self.device)
            x = x.squeeze().to(self.device)
            for t in range(x.shape[-1]):
                attn_weights = self.attn(hidden[-1])
                # attn_weights = self.attn(torch.cat([hidden[-1],x[:,:,t]],dim=1))
                # attn_output = torch.bmm(attn_weights,x[:,:,t].unsqueeze(1))
                attn_output = attn_weights.unsqueeze(1)*x[:,:,t].unsqueeze(1)
                attn_norms += torch.norm(attn_weights)
                gru_output,hidden  = self.gru(attn_output,hidden)
            y_hat = self.mlp(gru_output.squeeze())
            return y_hat
            # return y_hat.squeeze(), hidden, attn_norms     

        # x-> (batch_size, 1280/512, H, W)
        # x = x.max(dim=-1)[0].max(dim=-1)[0] # change it to mean
        if self.use_cbam:
            x = self.cbam(x)
        if self.use_pna:
            x = self.pna(x)
        x = torch.mean(x, dim=(-1, -2))
        x = self.final(x)  # -> (batch_size, num_classes)
        return x


def mixup_data(x, y, alpha):
    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def configureTorchDevice(cuda=torch.cuda.is_available()):
    """Configures PyTorch to use GPU and prints the same.

    Args:
        cuda (bool): To enable GPU, cuda is True, else false. If no value, will then check if GPU exists or not.  

    Returns:
        torch.device: PyTorch device, which can be either cpu or gpu
    """
    device = torch.device('cuda:0' if cuda else 'cpu')
    print('Device: ', device)
    return device


def getSampleRateString(sample_rate: int):
    """return sample rate in Khz in string form

    Args:
        sample_rate (int): sample rate in Hz

    Returns:
        str: string of sample rate in kHz
    """
    return f"{sample_rate/1000}k"


def dataSampleRateString(type: str, sample_rate: int):
    """Compute string name for the type of data and sample_rate

    Args:
        type (str): type/purpose of data
        sample_rate (int): sample rate of data in Hz

    Returns:
        str: string name for the type of data and sample_rate
    """
    return f"{type}_{getSampleRateString(sample_rate)}"
