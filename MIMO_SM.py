'''
@Author: Jr Ding
@File: MIMO_SM.py
@Time: 2023/2/17 9：03
@coding:utf-8
@brief:4×4 MIMO communication system simulation, research only spatial multiplexing(SM) and signal detection
'''

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# 系统参数
M = 4  # 发射天线数
N = 4  # 接收天线数
SNR_dB_list = []
BER_MMSE_list = []
BER_ZF_list = []

for SNR_dB in range(-20, 25, 5):   # 信噪比（dB）


    # 生成随机发送符号   10000个从0-3
    s = np.random.randint(0, M, 50000)

    # 将发送符号转换为向量形式  s_vec为4×1000的二维数组
    s_vec = np.zeros((M, 50000))
    for i in range(50000):
        s_vec[s[i], i] = 1

    # 生成随机信道矩阵
    H = np.random.randn(N, M) + 1j * np.random.randn(N, M)

    # 发送信号
    x = s_vec * 2

    # 接收信号
    n = np.sqrt(0.5 / (10 ** (SNR_dB / 10))) * (np.random.randn(N, 50000) + 1j * np.random.randn(N, 50000))
    y = H @ x + n    # @是矩阵乘法

    # ZF检测
    w_ZF = np.linalg.inv(H.conj().T @ H) @ H.conj().T
    s_hat_vec_ZF = w_ZF @ y

    # 最小均方误差（MMSE）检测
    H_inv = np.linalg.inv(H.conj().T @ H + 10 ** (-SNR_dB / 10) * np.eye(M))
    w_MMSE = H_inv @ H.conj().T
    s_hat_vec_MMSE = w_MMSE @ y

    # 将检测的符号向量转换为整数序列
    s_hat_ZF = np.zeros(50000, dtype=int)
    s_hat_MMSE = np.zeros(50000, dtype=int)
    for i in range(50000):
        s_hat_MMSE[i] = np.argmax(s_hat_vec_MMSE[:, i])
        s_hat_ZF[i] = np.argmax(s_hat_vec_ZF[:, i])

    # 计算误码率
    num_err_ZF = np.sum(s != s_hat_ZF)
    num_err_MMSE = np.sum(s != s_hat_MMSE)
    BER_ZF = num_err_ZF / 50000
    BER_MMSE = num_err_MMSE / 50000

    # 输出误码率
    print('信噪比 =',SNR_dB,'dB,', 'ZF误码率 =', BER_ZF, ',MMSE误码率 =', BER_MMSE)
    SNR_dB_list.append(SNR_dB)
    BER_MMSE_list.append(BER_MMSE)
    BER_ZF_list.append(BER_ZF)

plt.figure(1)
plt.semilogy(SNR_dB_list, BER_ZF_list, label='ZF')
plt.semilogy(SNR_dB_list, BER_MMSE_list, label='MMSE')
plt.scatter(SNR_dB_list, BER_ZF_list)
plt.scatter(SNR_dB_list, BER_MMSE_list)
plt.xlabel('SNR(dB)')
plt.ylabel('BER')
plt.title('4×4 MIMO')
plt.legend()
plt.show()


