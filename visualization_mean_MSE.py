import os
import numpy as np
import matplotlib.pyplot as plt

# 对 10 个 fold 的每个时间点求平均 MSE 并绘制折线图
base_dir = 'results/hcp_test_entire'
save_dir = os.path.join(base_dir, 'figures/mean_mse')
os.makedirs(save_dir, exist_ok=True)

all_mse = []
min_len = None

for i in range(1, 11):
    file_path = os.path.join(
        base_dir, 'all_sub_fold_' + str(i) + '_test_mse_entire_pred.npy'
    )
    if not os.path.exists(file_path):
        print('未找到文件: ' + file_path + '，已跳过该 fold')
        continue

    mse = np.load(file_path).squeeze()
    if mse.ndim != 1:
        print('文件维度不符合预期: ' + file_path + '，已跳过该 fold')
        continue

    if min_len is None or len(mse) < min_len:
        min_len = len(mse)
    all_mse.append(mse)

if not all_mse:
    raise RuntimeError('未加载到有效的 MSE 数据，请检查文件路径和文件内容')

# 对齐长度后按时间点求平均
all_mse = [m[:min_len] for m in all_mse]
all_mse = np.stack(all_mse, axis=0)
# print(all_mse.shape)
# exit()
mean_mse_by_time = np.mean(all_mse, axis=0)
time_indices = np.arange(50, 200)

plt.figure(figsize=(12, 8))
plt.plot(time_indices, mean_mse_by_time[:150])
plt.xlabel('Time Points')
plt.ylabel('mean MSE')
# plt.title('testset MSE')
plt.tight_layout()

save_path = os.path.join(save_dir, 'mean_mse_by_time.png')
plt.savefig(save_path, dpi=300)
plt.show()
