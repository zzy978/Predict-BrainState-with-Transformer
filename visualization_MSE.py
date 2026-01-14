import numpy as np
import matplotlib.pyplot as plt

# 加载一个 fold 的结果
for i in range(1, 11):
    mse = np.load('results/hcp_test_teaching/entire/all_sub_fold_' + str(i) + '_test_mse_entire_pred.npy')
    # shape: (1, T)

    mse = mse[0]  # 只有一个 test subject
    plt.figure(figsize=(12,8))
    plt.plot(mse)
    plt.xlabel('Prediction step')
    plt.ylabel('MSE')
    plt.title('Autoregressive Prediction Error')
    # plt.show()

    save_path = 'results/hcp_test_teaching/entire/mse_figures/'
    plt.savefig(save_path + 'fold_' + str(i) + '_autoregressive_prediction_error.png')