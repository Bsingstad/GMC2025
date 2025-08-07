#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import time
import joblib
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
from scipy import signal
import wfdb
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt
from torch.utils.data import WeightedRandomSampler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    PRETRAIN = True
    FINETUNE = False
    NEW_FS = 400
    TIME = 7  # seconds
    AUXILIARY = False
    #selected_leads = [0, 1, 2] + list(range(-6, 0))  # [-6, -5, -4, -3, -2, -1]
    selected_leads = np.arange(12)  # Use all 12 leads
    SOURCE = "# Source:"
    os.makedirs(model_folder, exist_ok=True)
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)
    records = sorted(records)

    pretrain_auxillary_labels = pd.read_csv(os.path.join("./", 'exams.csv'),dtype={'exam_id': str})


    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    #features = np.zeros((num_records, 6), dtype=np.float64)
    #X_data = np.zeros((num_records, 1000, 12), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)
    source_list = []
    record_list = [] 

    # Iterate over the records.
    for i in tqdm(range(num_records)):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')
        
        record = os.path.join(data_folder, records[i])
        record_list.append(record)
        labels[i] = load_label(record)
        source_list.append(get_source(record))

    
    record_list = np.asarray(record_list)
    source_list = np.asarray(source_list)
    record_list_stripped = [os.path.basename(record) for record in record_list]
    #record_list_stripped = [int(s) for s in record_list_stripped]
    #print("record_list_stripped:", record_list_stripped)
    #time.sleep(50)
    #records = [int(os.path.basename(s)) for s in records]
    #
    #records = [str(s) for s in records]

    #indices_pretrain = np.where(source_list == 'CODE-15%')[0]
    indices_pretrain = np.where((source_list == 'CODE-15%')|(source_list == 'SaMi-Trop')|(source_list == 'Athlete'))[0]
    indices_finetune = np.where((source_list == 'SaMi-Trop')|(source_list == 'PTB-XL')|(source_list == 'Athlete'))[0]
   
    # Train the models.
    if verbose:
        print('Training the model on the data...')
    
    if verbose:
        print('Pre-training the model...')

    temp_model = "./tempmodel/"
    temp_model_name = "temp_pretrain_model.weights.h5"
    os.makedirs(temp_model, exist_ok=True)

    record_list_stripped = np.asarray(record_list_stripped)

    record_list_stripped_pretrain = record_list_stripped[indices_pretrain]
    record_list_stripped_pretrain = [int(s) for s in record_list_stripped_pretrain]
    


    gpu_id = 0
    batch_size = 32
    lr = 1e-4
    weight_decay = 1e-5
    early_stop_lr = 1e-5
    Epochs = 5
    saved_dir = './eval/'
    os.makedirs('./eval/', exist_ok=True)

    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu")

    record_list_pretrain = record_list[indices_pretrain]
    labels_pretrain = labels[indices_pretrain]


    labels_pretrain = np.expand_dims(labels_pretrain, axis=1)
    print(labels_pretrain.shape)


    X_train, X_val, y_train, y_val = train_test_split(record_list_pretrain,labels_pretrain, test_size=0.15, random_state=42)
    
    n_classes = y_train.shape[1]
        #ECGdataset = LVEF_12lead_cls_Dataset()
    pth = './pretrained_model/12_lead_ECGFounder.pth'
    model = ft_12lead_ECGFounder(device, pth, n_classes,linear_prob=False)

    train_dataset = ECGDataset(X_train, y_train, selected_leads=selected_leads, new_fs=NEW_FS, time=TIME)
    sampler = make_balanced_sampler(y_train)
    trainloader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=sampler,
        num_workers=0
    )

    val_dataset = ECGDataset(X_val, y_val, selected_leads=selected_leads, new_fs=NEW_FS, time=TIME)
    valloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, mode='max', verbose=True)

    ### train model
    best_val_auroc = 0.
    step = 0
    current_lr = lr
    all_res = []
    pos_neg_counts = {}
    total_steps_per_epoch = len(trainloader)
    eval_steps = total_steps_per_epoch

    for epoch in range(Epochs):
        ### train
        for batch in tqdm(trainloader,desc='Training'):
            input_x, input_y = tuple(t.to(device) for t in batch)
            #input_y = input_y.squeeze(-1)  # My fix
            outputs = model(input_x)
            loss = criterion(outputs, input_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1


    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose=False):
    """
    Loads the fine-tuned PyTorch model based on the ft_12lead_ECGFounder architecture.
    """
    n_classes = 1  # Binary classification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 🔥 Path to the pretrained weights checkpoint
    pretrained_path = os.path.join(model_folder, "model.pt")
    if not os.path.isfile(pretrained_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {pretrained_path}")

    # 🔧 Initialize the model using your constructor
    model = ft_12lead_ECGFounder(device=device, pth=pretrained_path, n_classes=n_classes, linear_prob=True)

    if verbose:
        print(f"Model successfully loaded from: {pretrained_path}")

    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    NEW_FS = 400
    TIME = 7  # seconds
    #selected_leads = [0, 1, 2] + list(range(-6, 0))  # [-6, -5, -4, -3, -2, -1]
    selected_leads = np.arange(12)  # Use all 12 leads

 # Load ECG
    ecg, text = load_signals(record)
    fs = int(text["fs"])
    fs_ratio = NEW_FS / fs

    # Resample
    ecg_resamp = signal.resample(ecg, int(ecg.shape[0] * fs_ratio), axis=0)
    ecg_resamp = np.moveaxis(ecg_resamp, 0, -1)  # shape: (leads, time)

    # Select leads
    ecg_resamp = ecg_resamp[selected_leads, :]

    # Pad or truncate to fixed time window
    max_len = NEW_FS * TIME
    signal_len = ecg_resamp.shape[1]
    if signal_len < max_len:
        pad_width = max_len - signal_len
        ecg_resamp = np.pad(ecg_resamp, ((0, 0), (0, pad_width)), mode='constant')
    else:
        ecg_resamp = ecg_resamp[:, :max_len]

    # Reshape and convert to torch tensor
    input_tensor = torch.tensor(ecg_resamp, dtype=torch.float32).unsqueeze(0)  # shape: (1, leads, time)
    input_tensor = input_tensor.to(next(model.parameters()).device)

    # Run model
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

    binary_output = (probs > 0.5).astype(int)

    return binary_output, probs

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)
    
    one_hot_encoding_sex = np.zeros(3, dtype=bool)
    if sex == 'Female':
        one_hot_encoding_sex[0] = 1
    elif sex == 'Male':
        one_hot_encoding_sex[1] = 1
    else:
        one_hot_encoding_sex[2] = 1

    signal, fields = load_signals(record)

    # TO-DO: Update to compute per-lead features. Check lead order and update and use functions for reordering leads as needed.

    num_finite_samples = np.size(np.isfinite(signal))
    if num_finite_samples > 0:
        signal_mean = np.nanmean(signal)
    else:
        signal_mean = 0.0
    if num_finite_samples > 1:
        signal_std = np.nanstd(signal)
    else:
        signal_std = 0.0

    features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))

    return np.asarray(features, dtype=np.float32)

# Save your trained model.
def save_model(model_folder, model):
    """
    Save a PyTorch model to disk.

    Args:
        model_folder (str): Path to the folder to save the model.
        model (torch.nn.Module): Trained PyTorch model.
        filename (str): File name to save the model. Default is 'model.pt'.
    """
    filename="model.pt"
    os.makedirs(model_folder, exist_ok=True)
    path = os.path.join(model_folder, filename)
    torch.save(model.state_dict(), path)



def get_source(record):
    # Load the record
    text = load_header(record)
    # Extract the source information from the text header
    source = get_variable(text, "# Source:")
    return source[0] 



"""
a modularized deep neural network for 1-d signal data, pytorch version
 
Shenda Hong, Mar 2020
"""



class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.long))

    def __len__(self):
        return len(self.data)

class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding

    input: (n_sample, in_channels, n_length)
    output: (n_sample, out_channels, (n_length+stride-1)//stride)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        # print(net.shape)
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding

    params:
        kernel_size: kernel size
        stride: the stride of the window. Default value is kernel_size
    
    input: (n_sample, n_channel, n_length)
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        p = max(0, self.kernel_size - 1)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class BasicBlock(nn.Module):
    """
    Basic Block: 
        conv1 -> convk -> conv1

    params:
        in_channels: number of input channels
        out_channels: number of output channels
        ratio: ratio of channels to out_channels
        kernel_size: kernel window length
        stride: kernel step size
        groups: number of groups in convk
        downsample: whether downsample length
        use_bn: whether use batch_norm
        use_do: whether use dropout

    input: (n_sample, in_channels, n_length)
    output: (n_sample, out_channels, (n_length+stride-1)//stride)
    """
    def __init__(self, in_channels, out_channels, ratio, kernel_size, stride, groups, downsample, is_first_block=False, use_bn=True, use_do=True):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.downsample = downsample
        self.stride = stride if self.downsample else 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        self.middle_channels = int(self.out_channels * self.ratio)

        # the first conv, conv1
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.activation1 = Swish()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=self.in_channels, 
            out_channels=self.middle_channels, 
            kernel_size=1, 
            stride=1,
            groups=1)

        # the second conv, convk
        self.bn2 = nn.BatchNorm1d(self.middle_channels)
        self.activation2 = Swish()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=self.middle_channels, 
            out_channels=self.middle_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the third conv, conv1
        self.bn3 = nn.BatchNorm1d(self.middle_channels)
        self.activation3 = Swish()
        self.do3 = nn.Dropout(p=0.5)
        self.conv3 = MyConv1dPadSame(
            in_channels=self.middle_channels, 
            out_channels=self.out_channels, 
            kernel_size=1, 
            stride=1,
            groups=1)

        # Squeeze-and-Excitation
        r = 2
        self.se_fc1 = nn.Linear(self.out_channels, self.out_channels//r)
        self.se_fc2 = nn.Linear(self.out_channels//r, self.out_channels)
        self.se_activation = Swish()

        if self.downsample:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        out = x
        # the first conv, conv1
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.activation1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv, convk
        if self.use_bn:
            out = self.bn2(out)
        out = self.activation2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # the third conv, conv1
        if self.use_bn:
            out = self.bn3(out)
        out = self.activation3(out)
        if self.use_do:
            out = self.do3(out)
        out = self.conv3(out) # (n_sample, n_channel, n_length)

        # Squeeze-and-Excitation
        se = out.mean(-1) # (n_sample, n_channel)
        se = self.se_fc1(se)
        se = self.se_activation(se)
        se = self.se_fc2(se)
        se = torch.sigmoid(se) # (n_sample, n_channel)
        out = torch.einsum('abc,ab->abc', out, se)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out

class BasicStage(nn.Module):
    """
    Basic Stage:
        block_1 -> block_2 -> ... -> block_M
    """
    def __init__(self, in_channels, out_channels, ratio, kernel_size, stride, groups, i_stage, m_blocks, use_bn=True, use_do=True, verbose=False):
        super(BasicStage, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.i_stage = i_stage
        self.m_blocks = m_blocks
        self.use_bn = use_bn
        self.use_do = use_do
        self.verbose = verbose

        self.block_list = nn.ModuleList()
        for i_block in range(self.m_blocks):
            
            # first block
            if self.i_stage == 0 and i_block == 0:
                self.is_first_block = True
            else:
                self.is_first_block = False
            # downsample, stride, input
            if i_block == 0:
                self.downsample = True
                self.stride = stride
                self.tmp_in_channels = self.in_channels
            else:
                self.downsample = False
                self.stride = 1
                self.tmp_in_channels = self.out_channels
            
            # build block
            tmp_block = BasicBlock(
                in_channels=self.tmp_in_channels, 
                out_channels=self.out_channels, 
                ratio=self.ratio, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                groups=self.groups, 
                downsample=self.downsample, 
                is_first_block=self.is_first_block,
                use_bn=self.use_bn, 
                use_do=self.use_do)
            self.block_list.append(tmp_block)

    def forward(self, x):

        out = x

        for i_block in range(self.m_blocks):
            net = self.block_list[i_block]
            out = net(out)
            if self.verbose:
                print('stage: {}, block: {}, in_channels: {}, out_channels: {}, outshape: {}'.format(self.i_stage, i_block, net.in_channels, net.out_channels, list(out.shape)))
                print('stage: {}, block: {}, conv1: {}->{} k={} s={} C={}'.format(self.i_stage, i_block, net.conv1.in_channels, net.conv1.out_channels, net.conv1.kernel_size, net.conv1.stride, net.conv1.groups))
                print('stage: {}, block: {}, convk: {}->{} k={} s={} C={}'.format(self.i_stage, i_block, net.conv2.in_channels, net.conv2.out_channels, net.conv2.kernel_size, net.conv2.stride, net.conv2.groups))
                print('stage: {}, block: {}, conv1: {}->{} k={} s={} C={}'.format(self.i_stage, i_block, net.conv3.in_channels, net.conv3.out_channels, net.conv3.kernel_size, net.conv3.stride, net.conv3.groups))

        return out

class Net1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    params:
        in_channels
        base_filters
        filter_list: list, filters for each stage
        m_blocks_list: list, number of blocks of each stage
        kernel_size
        stride
        groups_width
        n_stages
        n_classes
        use_bn
        use_do

    """

    def __init__(self, in_channels, base_filters, ratio, filter_list, m_blocks_list, kernel_size, stride, groups_width, n_classes, use_bn=True, use_do=True, return_features=False, verbose=False):
        super(Net1D, self).__init__()
        
        self.in_channels = in_channels
        self.base_filters = base_filters
        self.ratio = ratio
        self.filter_list = filter_list
        self.m_blocks_list = m_blocks_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups_width = groups_width
        self.n_stages = len(filter_list)
        self.n_classes = n_classes
        self.use_bn = use_bn
        self.use_do = use_do
        self.return_features = return_features
        self.verbose = verbose

        # first conv
        self.first_conv = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=self.base_filters, 
            kernel_size=self.kernel_size, 
            stride=2)
        self.first_bn = nn.BatchNorm1d(base_filters)
        self.first_activation = Swish()

        # stages
        self.stage_list = nn.ModuleList()
        in_channels = self.base_filters
        for i_stage in range(self.n_stages):

            out_channels = self.filter_list[i_stage]
            m_blocks = self.m_blocks_list[i_stage]
            tmp_stage = BasicStage(
                in_channels=in_channels, 
                out_channels=out_channels, 
                ratio=self.ratio, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                groups=out_channels//self.groups_width, 
                i_stage=i_stage,
                m_blocks=m_blocks, 
                use_bn=self.use_bn, 
                use_do=self.use_do, 
                verbose=self.verbose)
            self.stage_list.append(tmp_stage)
            in_channels = out_channels

        # final prediction
        self.dense = nn.Linear(in_channels, n_classes)
    """        
    def forward(self, x):
        
        out = x
        
        # first conv
        out = self.first_conv(out)
        if self.use_bn:
            out = self.first_bn(out)
        out = self.first_activation(out)
        
        # stages
        for i_stage in range(self.n_stages):
            net = self.stage_list[i_stage]
            out = net(out)

        # final prediction
        deep_features = out.mean(-1)
        out = self.dense(deep_features)

        if self.return_features:
            return out, deep_features
        else:
            return out
    """
    def forward(self, x):
        out = x
    
        # first conv
        out = self.first_conv(out)
        if self.use_bn:
            out = self.first_bn(out)
        out = self.first_activation(out)
    
        # stages
        for i_stage in range(self.n_stages):
            net = self.stage_list[i_stage]
            out = net(out)
    
        # final prediction
        deep_features = out.mean(-1)
    
        # ✅ Move features to same device as the dense layer (GPU)
        deep_features = deep_features.to(self.dense.weight.device)
    
        out = self.dense(deep_features)
    
        if self.return_features:
            return out, deep_features
        else:
            return out


def ft_12lead_ECGFounder(device, pth, n_classes, linear_prob=False):
    model = Net1D(
        in_channels=12, 
        base_filters=64,
        ratio=1, 
        filter_list=[64,160,160,400,400,1024,1024],
        m_blocks_list=[2,2,2,3,3,4,4],
        kernel_size=16, 
        stride=2, 
        groups_width=16,
        verbose=False, 
        use_bn=False,
        use_do=False,
        n_classes=n_classes
    )

    # Load checkpoint on CPU to avoid large transfer
    checkpoint = torch.load(pth, map_location='cpu')
        # If checkpoint contains 'state_dict', extract it, else assume checkpoint is the state_dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    #state_dict = checkpoint['state_dict']

    # Ignore last layer weights
    #state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')}
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')}


    model.load_state_dict(filtered_state_dict, strict=False)

    # Replace last layer and move full model to GPU
    model.dense = nn.Linear(model.dense.in_features, n_classes)

    # Freeze all layers except the last
    if linear_prob:
        for name, param in model.named_parameters():
            if 'dense' not in name:
                param.requires_grad = False

    # ✅ Now move the WHOLE model to GPU
    model = model.to(device)

    return model




class ECGDataset(Dataset):
    def __init__(self, record_paths, labels, selected_leads=None, new_fs=400, time=7):
        """
        record_paths: list of paths to ECG records
        labels: numpy array or list of label vectors (e.g., binary/multi-label)
        selected_leads: list of lead indices to keep, e.g., [0, 1, 2, -6, -5, -4, -3, -2, -1]
        """
        self.record_paths = record_paths
        self.labels = labels
        self.selected_leads = selected_leads
        self.new_fs = new_fs
        self.time = time
        self.length = int(self.new_fs * self.time)

    def __len__(self):
        return len(self.record_paths)

    def __getitem__(self, idx):
        record = self.record_paths[idx]
        label = self.labels[idx]

        ecg, meta = load_signals(record)  # load_signals must return (signal_array, metadata_dict)
        fs = int(meta["fs"])
        fs_ratio = self.new_fs / fs

        # Resample if needed
        if self.new_fs != fs:
            ecg_resamp = signal.resample(ecg, int(ecg.shape[0] * fs_ratio), axis=0)
        else:
            ecg_resamp = ecg

        # Transpose to shape [channels, time]
        ecg_resamp = np.moveaxis(ecg_resamp, 0, -1)

        # Select leads
        selected = ecg_resamp[self.selected_leads]

        # Pad or truncate to target length
        padded = np.zeros((len(self.selected_leads), self.length), dtype=np.float32)
        seq_len = min(selected.shape[1], self.length)
        padded[:, :seq_len] = selected[:, :seq_len]

        # Return in (channels, time) format (B, C, T)
        return torch.tensor(padded, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def make_balanced_sampler(labels):
    """
    Args:
        labels: shape (N,) or (N, 1), binary or bool
    Returns:
        WeightedRandomSampler for balanced DataLoader
    """
    labels = np.array(labels)

    # If 2D (e.g., shape [N, 1]), flatten it
    if labels.ndim > 1:
        labels = labels.squeeze()

    # If labels are boolean, cast to int
    if labels.dtype == bool:
        labels = labels.astype(int)

    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]
    
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
