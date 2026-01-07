import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def pearson_corr(x, y):
    """计算两个一维向量的 Pearson 相关系数"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom == 0:
        return np.nan
    return float(np.sum(x * y) / denom)


def load_subject_list(true_dir):
    """优先读取测试划分文件，确保 fold 与被试对应"""
    # pickle_path = 'testpy_subjects.pickle'
    # if os.path.exists(pickle_path):
    #     with open(pickle_path, 'rb') as f:
    #         test_sub_split = pickle.load(f)
    #     subject_list = []
    #     for item in test_sub_split:
    #         if isinstance(item, (list, tuple)) and len(item) > 0:
    #             subject_list.append(item[0])
    #         elif isinstance(item, str):
    #             subject_list.append(item)
    #     return subject_list

    if os.path.isdir(true_dir):
        files = [f for f in os.listdir(true_dir) if f.endswith('.npy')]
        files.sort()
        return files
    return []


# 路径设置
pred_dir = 'new_models/test_entire'
true_dir = 'dataset_test'
save_dir = os.path.join(pred_dir, 'figures')
os.makedirs(save_dir, exist_ok=True)

subject_list = load_subject_list(true_dir)
if len(subject_list) < 10:
    raise RuntimeError('可用的被试文件少于 10 个，请检查 dataset_test 或 testpy_subjects.pickle')

all_corr = []
min_len = None

for i in range(1, 11):
    pred_path = os.path.join(pred_dir, 'all_sub_fold_' + str(i) + '_fmri_predicted.npy')
    true_path = os.path.join(true_dir, subject_list[i - 1])

    if not os.path.exists(pred_path):
        print('未找到预测文件: ' + pred_path + '，已跳过该 fold')
        continue
    if not os.path.exists(true_path):
        print('未找到真实文件: ' + true_path + '，已跳过该 fold')
        continue

    pred = np.load(pred_path).squeeze()
    true = np.load(true_path).squeeze()

    if pred.ndim != 2 or true.ndim != 2:
        print('文件维度不符合预期: ' + pred_path + ' 或 ' + true_path + '，已跳过该 fold')
        continue

    if pred.shape != true.shape:
        if pred.T.shape == true.shape:
            pred = pred.T
        elif true.T.shape == pred.shape:
            true = true.T
        else:
            print('形状不匹配: ' + pred_path + ' vs ' + true_path + '，已跳过该 fold')
            continue

    time_len = min(pred.shape[0], true.shape[0])
    if min_len is None or time_len < min_len:
        min_len = time_len

    fold_corr = []
    for t in range(time_len):
        corr = pearson_corr(pred[t], true[t])
        fold_corr.append(corr)
    all_corr.append(np.array(fold_corr, dtype=np.float64))

    print('Fold ' + str(i) + ' 使用被试文件: ' + subject_list[i - 1])

if not all_corr:
    raise RuntimeError('未加载到有效的数据，请检查文件路径与文件内容')

# 对齐时间长度并对 10 个 fold 求平均
all_corr = [c[:min_len] for c in all_corr]
all_corr = np.stack(all_corr, axis=0)
mean_corr_by_time = np.nanmean(all_corr, axis=0)

time_idx = np.arange(1, min_len + 1)
plt.figure(figsize=(6, 4))
plt.plot(time_idx, mean_corr_by_time)
plt.xlabel('Time Points')
plt.ylabel('Correlation (Pearson)')
# plt.title('10 对文件在每个时间点的平均 Pearson 相关系数')
plt.tight_layout()

save_path = os.path.join(save_dir, 'mean_pearson_by_time.png')
plt.savefig(save_path, dpi=300)
plt.show()
