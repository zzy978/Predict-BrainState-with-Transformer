import os
import glob
import matplotlib.pyplot as plt # 绘图库
import numpy as np # 处理数值数组
import pandas as pd
import random
import torch
from torch.utils.data import DataLoader,random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from fmri_dataset import rfMRIDataset
from model import *
from tqdm import tqdm
import pickle

def mse_calc(x, x_hat):
    reproduction_loss = nn.functional.mse_loss(x_hat, x)
    return reproduction_loss

def se_calc(x, x_hat):
    x1 = x.cpu()
    x2 = x_hat.cpu()
    x1 = x1.numpy()
    x2 = x2.numpy()
    se = (x1 - x2)**2
    return se

def gen_square_subsequent_mask(sz, device):
    # 因果掩码：[sz, sz]
    return torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)

################################################################
""" 参数 """
folds = 10
dir = 'dataset_hcp_test/'
max_window_size = 50
window_size = 30
pred_len = 20
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
shuffle = False
batch_size = 1
################################################################
if shuffle:
    print('Shuffle the input time series')
# 获取所有被试编号
all_sub = os.listdir(dir)
all_sub.sort()

# 10 折划分
with open('testpy_hcp_subjects.pickle', 'rb') as file:
    test_sub_split = pickle.load(file)
for fold in range(folds):
    test_sub = test_sub_split[fold]
    print(f'Fold {fold+1}')
    print(test_sub)
    print(len(test_sub))

    # 加载数据
    test_data = rfMRIDataset(dir, test_sub, window_size, max_window_size, pred_len=pred_len)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    # 测试
    model = torch.load('new_models/hcp_epo10_win30_teaching/hcp_transformer_fold_'+str(fold+1)+'_epo-10_win-30.pth', map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    test_mse_first = []
    test_mse_aux = []
    test_mse_weighted = []

    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_dataloader)):
            if shuffle:
                data = data.squeeze()
                idx = torch.randperm(data.size(0))
                data = data[idx, :]
                data = data.unsqueeze(0)

            # 准备编码器和解码器的输入
            # print(data.shape)
            # exit()
            encoder_input = data.to(device)
            decoder_input = data[:, -1, :].unsqueeze(1).to(device) # 增加单时间点维度
            # 确保数据类型为 float64
            encoder_input = encoder_input.float()
            decoder_input = decoder_input.float()
            # Transformer 输出
            target = target.to(device)
            target = target.float()
            tgt_in = torch.cat([decoder_input, target[:, :-1, :]], dim=1)
            tgt_mask = gen_square_subsequent_mask(pred_len, device)
            pred = model(src=encoder_input, tgt=tgt_in, tgt_mask=tgt_mask)  # [B, K, ROI]
            # 评价指标：第 1 步为主，后续为辅
            first_loss = F.mse_loss(pred[:, 0, :], target[:, 0, :]).item()
            if pred_len > 1:
                aux_loss = F.mse_loss(pred[:, 1:, :], target[:, 1:, :]).item()
            else:
                aux_loss = 0.0
            weighted = 0.7 * first_loss + 0.3 * aux_loss
            test_mse_first.append(first_loss)
            test_mse_aux.append(aux_loss)
            test_mse_weighted.append(weighted)
            
    test_mse_first = np.array(test_mse_first)
    test_mse_aux = np.array(test_mse_aux)
    test_mse_weighted = np.array(test_mse_weighted)
    print('Test First-step MSE: ', np.mean(test_mse_first))
    print('Test Aux MSE: ', np.mean(test_mse_aux))
    print('Test Weighted MSE: ', np.mean(test_mse_weighted))
    print(test_mse_weighted.shape)
    if shuffle:
        # np.save('new_models/test_teacherforcing/all_sub_fold_'+str(fold+1)+'_test_mse_first_shuffled.npy', test_mse_first)
        # np.save('new_models/test_teacherforcing/all_sub_fold_'+str(fold+1)+'_test_mse_aux_shuffled.npy', test_mse_aux)
        # np.save('new_models/test_teacherforcing/all_sub_fold_'+str(fold+1)+'_test_mse_weighted_shuffled.npy', test_mse_weighted)
        # 兼容旧命名：保持 weighted 作为 test_mse
        np.save('results/hcp_test_teaching/single/all_sub_fold_'+str(fold+1)+'_test_mse_shuffled.npy', test_mse_weighted)
    else:
        # np.save('new_models/test_teacherforcing/all_sub_fold_'+str(fold+1)+'_test_mse_first.npy', test_mse_first)
        # np.save('new_models/test_teacherforcing/all_sub_fold_'+str(fold+1)+'_test_mse_aux.npy', test_mse_aux)
        # np.save('new_models/test_teacherforcing/all_sub_fold_'+str(fold+1)+'_test_mse_weighted.npy', test_mse_weighted)
        # 兼容旧命名：保持 weighted 作为 test_mse
        np.save('results/hcp_test_teaching/single/all_sub_fold_'+str(fold+1)+'_test_mse.npy', test_mse_weighted)