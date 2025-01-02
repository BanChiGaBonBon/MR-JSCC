import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft,ifft
import torch
# 创建一个时间向量
# t = np.linspace(0, 8/500, 8, endpoint=False)
# 创建一个包含两个频率成分的信号
# 这里我们使用正弦波作为示例
# t = np.linspace(0, 1, 64, endpoint=False)

# f1 = 4  # 频率1为10Hz
# f2 = 0    # 频率2为20Hz
# s = np.reshape(np.arange(64)) 
def generate_bits(n):
    """生成随机比特流"""
    return np.random.randint(0, 2, n)

def qpsk_modulation(bits):
    """QPSK调制"""
    symbols = np.zeros((len(bits) // 2), dtype=complex)
    for i in range(0, len(bits), 2):
        real = 1 if bits[i] else -1
        imag = 1 if bits[i+1] else -1
        symbols[i//2] = (real + 1j * imag) / np.sqrt(2)  # 归一化
    return symbols

s = np.arange(24)
s = torch.tensor(s).view(3,2,4)
s2 = s.permute(0,2,1).permute(1,2,0)
print(s)
print(s2)
print(s.view(6,2,2))
print(s2.contiguous().view(6,2,2))



N = 1
# s = np.arange(M*N)
# s = s-2j*s
# s = np.reshape(s,(N,M))
M = 8
s = np.zeros(M)
s[2] = 1

# bits = generate_bits(M*N*2)
# s = qpsk_modulation(bits)
# s = np.reshape(s,(N,M))
# signal_otfs = np.reshape(ifft(s,axis=0),N * M) /M
signal_ofdm = np.reshape(ifft(s),N * M) 

# signal = ifft(s,axis=axis)


v=100
c=3e8
fc = 3.6e9
subspace = 15e3

fs = subspace * M
t = np.arange(N*M)/fs
fd = (fc)*v/c
fd = 1000
# signal_otfs= (signal_otfs) * np.exp(2j*np.pi*fd*t)
signal_ofdm= (signal_ofdm) * np.exp(2j*np.pi*fd*t)
# fft_result = fft(signal)

# otfs_result = np.reshape(fft(np.reshape(signal_otfs,(N,M)),axis=0),N * M) *M
l = 8
ofdm_result = (fft(np.reshape(signal_ofdm,(N,M)), l)) 

# # print(ofdm_result)
# fft_result = signal
# print(fft_result)
# # 计算频率轴
# plt.figure(figsize=(8, 8))
# plt.scatter(s.real, s.imag, color='blue', label='ori')
# plt.scatter(otfs_result.real, otfs_result.imag, color='red', alpha=0.5, label='otfs')
# plt.scatter(ofdm_result.real, ofdm_result.imag, color='green', alpha=0.5, label='ofdm')
# plt.title('QPSK 调制的发送和接收符号')
# plt.xlabel('实部')
# plt.ylabel('虚部')
# plt.grid(True)
# plt.axis('equal')
# plt.legend()
# freq = np.linspace(0, len(signal_ofdm), len(signal_ofdm), endpoint=False)
freq = np.linspace(0, l, l, endpoint=False)
# # 绘制原始信号和频谱图
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1)
# plt.plot(t, signal)


# plt.scatter(freq, otfs_result.real,linewidths=0.01,c='black') 
plt.scatter(freq, ofdm_result.real,linewidths=0.1) 
# plt.scatter(freq, s.real,linewidths=0.1) 
plt.title('real Signal')
plt.xlabel('f')
plt.ylabel('Amplitude')
plt.subplot(1, 3, 2)

# plt.scatter(freq, otfs_result.imag,linewidths=0.1,c='black')  # 只绘制一半的频谱图，因为频谱是对称的
plt.scatter(freq, ofdm_result.imag,linewidths=0.1)
# plt.scatter(freq, s.imag,linewidths=0.1)
# for i in range(len(freq)):
#     plt.text(freq[i], np.abs(fft_result)[i], f'({freq[i]}, {np.abs(fft_result)[i]})', fontsize=6, ha='center',va='bottom')
# plt.plot(freq, np.abs(fft_result)) 
plt.title('imag Signal')
plt.xlabel('f')
plt.ylabel('Amplitude')

plt.subplot(1, 3, 3)

# plt.scatter(freq, np.abs(otfs_result),linewidths=0.1,c='black')
plt.scatter(freq, np.abs(ofdm_result),linewidths=0.1)
# plt.scatter(freq, np.abs(s),linewidths=0.1)
plt.title('abs')
plt.xlabel('f')
plt.ylabel('Amplitude')

plt.tight_layout()

plt.savefig('t.png')

print("fd=",fd)