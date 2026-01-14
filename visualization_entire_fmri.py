import os
import numpy as np
import matplotlib.pyplot as plt

base_dir = 'results/hcp_test_teaching/entire/'
save_dir = os.path.join(base_dir, 'pred_figures/')
# os.makedirs(save_dir, exist_ok=True)

def load_subject_list(true_dir):
    """优先读取测试划分文件，确保 fold 与被试对应"""
    if os.path.isdir(true_dir):
        files = [f for f in os.listdir(true_dir) if f.endswith('.npy')]
        files.sort()
        return files
    return []

true_dir = 'dataset_hcp_test'
subject_list = load_subject_list(true_dir)

for i in range(1,11):
    file_path = os.path.join(
        base_dir, 'all_sub_fold_' + str(i) + '_fmri_predicted.npy'
    )
    true_path = os.path.join(true_dir, subject_list[i - 1])

    pred = np.load(file_path)
    pred = pred.squeeze()
    # print(pred.shape)
    true = np.load(true_path)

    roi = 10  # 随便挑一个 ROI

    plt.figure(figsize=(12,8))
    plt.plot(true[:, roi], label='True')
    plt.plot(pred[:, roi], label='Predicted')
    plt.axvline(x=50, color='r', linestyle='--', label='Prediction starts')
    plt.legend()
    plt.title(f'ROI {roi} Time Series')
    plt.savefig(os.path.join(save_dir, f'fold_{i}_roi_{roi}_timeseries.png'))

