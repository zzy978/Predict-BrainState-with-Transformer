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
dir = 'dataset_hcp_test/'
folds = 10
window_size = 30
max_window_size = 50
decode_len = 1  # 生成时只解码 1 步（训练时 K=20）
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
## 模型参数
batch_size = 1
################################################################

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

    # 获取所有文件
    files = []
    for sub in test_sub: # 遍历被试编号
        files += [dir + sub]
        
    # for sub in range(len(test_sub)): # 遍历被试编号
        # sub_files = glob.glob(dir+sub+'/*REST*_p.npy') # 获取子目录文件名
        # sub_files.sort() # 排序
        # files += test_sub[sub] # 加入列表
    # print(f'测试文件: {files}')
    # exit()

    # 加载模型
    model = torch.load('new_models/hcp_epo10_win30_teaching/hcp_transformer_fold_'+str(fold+1)+'_epo-10_win-30.pth', map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    test_mse = []
    test_mse_first = []
    test_mse_aux = []
    test_mse_weighted = []
    test_regional_mse = []
    all_ses_data_predicted = []
    print("测试文件:", files)
    with torch.no_grad():
        total_steps = sum(np.load(f).shape[0] - max_window_size for f in files)
        progress = tqdm(total=total_steps)
        for f in files:
            ses_mse = []
            ses_mse_first = []
            ses_mse_aux = []
            ses_mse_weighted = []
            ses_regional_mse = []
            i = max_window_size
            f_data = np.load(f)
            data_for_test = f_data[0:max_window_size, :]
            total_len = f_data.shape[0]
            while i < total_len:
                step_len = decode_len
                data = data_for_test[i-window_size:i, :]
                data = torch.tensor(data).unsqueeze(0)
                target = f_data[i:i+step_len, :]
                target = torch.tensor(target).unsqueeze(0)
                # 准备编码器和解码器的输入
                encoder_input = data.to(device)
                decoder_input = data[:, -1, :].unsqueeze(1).to(device) # 增加单时间点维度
                # 确保数据类型为 float64
                encoder_input = encoder_input.float()
                decoder_input = decoder_input.float()
                # Transformer 输出
                target = target.to(device)
                target = target.float()
                tgt_in = decoder_input
                tgt_mask = gen_square_subsequent_mask(step_len, device)
                pred = model(src=encoder_input, tgt=tgt_in, tgt_mask=tgt_mask)  # [1, 1, 脑区]
                # 逐步 MSE（用于兼容旧结果）
                step_mse = F.mse_loss(pred, target, reduction="none").mean(dim=2).squeeze(0)  # [1]
                ses_mse.extend(step_mse.detach().cpu().tolist())
                # 区域级误差（可选）
                se = se_calc(target, pred)
                ses_regional_mse.append(se)
                # 评价指标：第 1 步为主，后续为辅
                first_loss = step_mse[0].item()
                aux_loss = 0.0
                weighted = first_loss
                ses_mse_first.append(first_loss)
                ses_mse_aux.append(aux_loss)
                ses_mse_weighted.append(weighted)
                # 将预测时间点加入到序列中
                pred_np = pred.squeeze(0).detach().cpu().numpy()  # [1, 脑区]
                data_for_test = np.concatenate((data_for_test, pred_np), axis=0)
                i += step_len
                progress.update(step_len)
            test_mse.append(ses_mse)
            test_regional_mse.append(ses_regional_mse) 
            test_mse_first.append(ses_mse_first)
            test_mse_aux.append(ses_mse_aux)
            test_mse_weighted.append(ses_mse_weighted)
            all_ses_data_predicted.append(data_for_test)
    progress.close() 

    test_mse = np.array(test_mse)
    test_regional_mse = np.array(test_regional_mse)
    test_mse_first = np.array(test_mse_first)
    test_mse_aux = np.array(test_mse_aux)
    test_mse_weighted = np.array(test_mse_weighted)
    all_ses_data_predicted = np.array(all_ses_data_predicted)
    print('Test First-step MSE: ', np.mean(test_mse_first))
    print('Test Aux MSE: ', np.mean(test_mse_aux))
    print('Test Weighted MSE: ', np.mean(test_mse_weighted))
    print(test_mse.shape)
    # print(test_regional_mse.shape)
    # print(test_mse_weighted.shape)
    # print(all_ses_data_predicted.shape)
    np.save('results/hcp_test_teaching/entire/all_sub_fold_'+str(fold+1)+'_test_mse_entire_pred.npy', test_mse)
    # np.save('new_models/test_entire/all_sub_fold_'+str(fold+1)+'_test_mse_first_entire_pred.npy', test_mse_first)
    # np.save('new_models/test_entire/all_sub_fold_'+str(fold+1)+'_test_mse_aux_entire_pred.npy', test_mse_aux)
    # np.save('new_models/test_entire/all_sub_fold_'+str(fold+1)+'_test_mse_weighted_entire_pred.npy', test_mse_weighted)
    # # np.save('new_models/test/fold_'+str(fold+1)+'_test_regional_se_entire_pred.npy', test_regional_mse)
    np.save('results/hcp_test_teaching/entire/all_sub_fold_'+str(fold+1)+'_fmri_predicted.npy', all_ses_data_predicted)