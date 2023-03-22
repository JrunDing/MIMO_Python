'''
@Author: Jr Ding
@File: MIMO_STC.py
@Time: 2023/2/18 14：22
@coding:utf-8
@brief:2×1 MIMO STBC coding, mainly about Alamouti coding.
'''



import random
import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt


# 设置参数
M = 2 # 发送天线数
N = 1 # 接收天线数
noOfSamples = 20000  # 共 5000×2bits，发射5000×2次
SNRdB_list = range(0, 11) # 信噪比范围dB
H = [] # 存储信道矩阵
ber_2by1 = [None] * len(SNRdB_list)
err2by1 = []
ep2by1 = []



for SNRdB in SNRdB_list:
    # 噪声参数
    SNR = 10.0 ** (SNRdB / 10.0)  # 信噪比非dB
    noise_mean = 0  # 噪声均值
    noise_std = 1 / sqrt(2 * SNR)  # 噪声标准差   噪声方差 = 1/(2*SNR)

    no_errors_2by1 = 0  # 2×1 Alamouti编码的错误比特数
    # 一共发送5000次数据
    for sample in range(0, noOfSamples): # 每一组数据2bit 2天线
        N = []  # 存储噪声矩阵
        tx_bit1 = random.randint(0, 1)  # bit1
        tx_bit2 = random.randint(0, 1)  # bit2
        # BPSK调制
        tx_symbol1 = (2 * tx_bit1 - 1)+0j
        tx_symbol2 = (2 * tx_bit2 - 1)+0j
        tx_symbols = np.array([tx_symbol1, tx_symbol2])  # 发送符号向量 ndarray

        # 2×1 MIMO 信道矩阵
        for j in range(2):
            hxx = np.around(np.random.randn() - np.random.randn() * 1j, 5)  # hxx是h11, h12等 保留5位小数
            H.append(hxx)
            nxx = np.random.normal(loc=noise_mean, scale=noise_std, size=None) - np.random.normal(loc=noise_mean,
                                                                                                  scale=noise_std,
                                                                                                  size=None) * 1j
            N.append(nxx)

        # Alamouti Coding 信道矩阵
        H_2by1 = np.array([[H[0], H[1]], [H[1].conj(), -H[0].conj()]])
        # 取噪声
        n = np.array(N)
        Noise_2by1 = [n[0], n[1].conj()]  # noise vector for 2by1

        rx_symbols_2by1 = np.add((np.dot(H_2by1, (np.dot((1 / sqrt(2)), tx_symbols)))), Noise_2by1)

        # ZF线性检测
        symbols_est_2by1 = (np.linalg.inv(H_2by1.conj().T @ H_2by1) @ H_2by1.conj().T) @ rx_symbols_2by1 # ZF检测

        det_symbol1_2by1 = symbols_est_2by1[0]
        det_symbol2_2by1 = symbols_est_2by1[1]

        # 解调
        det_bit1_2by1 = (1 if (det_symbol1_2by1.real >= 0) else 0)
        det_bit2_2by1 = (1 if (det_symbol2_2by1.real >= 0) else 0)

        #print('!!!!!!!!!!!!')
        #print(tx_bit1)
        #print(det_bit1_2by1)
        #print(tx_bit2)
        #print(det_bit2_2by1)
        no_errors_2by1 += (1 if tx_bit1 != det_bit1_2by1 else 0) + (1 if(tx_bit2 != det_bit2_2by1) else 0)


    ber_2by1[SNRdB] = 1.0 * no_errors_2by1 / (2 * noOfSamples) # 指定信噪比下误码率

    err2by1.append(no_errors_2by1)
    ep2by1.append(ber_2by1[SNRdB])

    print("SNRdB:", SNRdB)
    print("Numbder of errors:", no_errors_2by1)
    print("Error probability:", ber_2by1[SNRdB])



# 画图
plt.figure(1)
plt.semilogy(SNRdB_list, ep2by1)
plt.scatter(SNRdB_list, ep2by1)
plt.title('Alamouti coding')
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.show()
