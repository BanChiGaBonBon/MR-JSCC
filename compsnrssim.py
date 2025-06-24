import matplotlib.pyplot as plt
import numpy as np
# 示例数据
d1 = '0630s32sv0snr'

# data1 =dict(np.load('%s.npy'%d1,allow_pickle=True).tolist())
data1 = {0:24.49142956756532,10:24.497510411615856,20:13.834595401628167}

d2 = 'S25.0316s16m30resSSIM'

data2 = np.load('%s.npy'%d2,allow_pickle=True).tolist()


d3 = 'S25.0316s16m30tcmSSIM'
# d3 = '0724s32resv50'
data3 = np.load('%s.npy'%d3,allow_pickle=True).tolist()

d4 = 'S25.0316s16m30ceeqotfscmSSIM'
data4 = np.load('%s.npy'%d4,allow_pickle=True).tolist()

d5 = 'S25.0406s16m30lstmcmSSIM'
data5 = np.load('%s.npy'%d5,allow_pickle=True).tolist()

d6 = 'S25.0405s16m30ccmSSIM'
data6 = np.load('%s.npy'%d6,allow_pickle=True).tolist()

d7 = 'S25.0316s16m30cmSSIM'
data7 = np.load('%s.npy'%d7,allow_pickle=True).tolist()

data8 = np.load('S75.0316s16m30resSSIM.npy',allow_pickle=True).tolist()
data9 = np.load('S75.0316s16m30tcmSSIM.npy',allow_pickle=True).tolist()
data10 = np.load('S75.0316s16m30ceeqotfscmSSIM.npy',allow_pickle=True).tolist()
data11= np.load('S75.0406s16m30lstmcmSSIM.npy',allow_pickle=True).tolist()
data12 = np.load('S75.0405s16m30ccmSSIM.npy',allow_pickle=True).tolist()
data13 = np.load('S75.0316s16m30cmSSIM.npy',allow_pickle=True).tolist()

# data14 = np.load('q470.525snr.npy',allow_pickle=True).tolist()
data14={"-4": 0, "-2": 0, "0": 0, "2": 0.12547521225788905, "4": 0.3520183046787849, "6": 0.5713105689265322, "8": 0.6647750491518, "10": 0.6758358274876127, "12": 0.6840220384768682, "14": 0.6741057500059358, "16": 0.6743164498229386}# 创建时间点（x轴）
time_points = range(0,110,10)

# 创建图形和轴
fig, ax = plt.subplots()

# 绘制第一组数据
#  # 使用圆圈标记数据点
ax.plot(list(data3.keys())[:11], list(data3.values())[:11], label='MR-JSCC', marker='*',color='green')  # 使用方形标记数据点
# ax.plot(list(data3.keys())[:11], list(data9.values())[:11], label='MR-JSCC $V_{test}=75m/s$', marker='*',color='green',ls='--')  # 使用方形标记数据点

# ax.plot(list(data1.keys()), list(data1.values()), label='BPG+1/2LDPC+QPSK+OFDM', marker='.') 
# ax.plot(list(data8.keys()), list(data8.values()), label='BPG+3/4LDPC+QPSK+OTFS', marker='.') 



ax.plot(list(data2.keys())[:11], list(data2.values())[:11], label='RES-JSCC', marker='v',color='blue')  # 使用方形标记数据点
# ax.plot(list(data8.keys())[:11], list(data8.values())[:11], label='RES-JSCC $V_{test}=75m/s$', marker='v',color='blue',ls='--')  # 使用方形标记数据点

ax.plot(list(data5.keys())[:11], list(data5.values())[:11], label='LSTM-JSCC', marker='o',color='yellow')  # 使用方形标记数据点
# ax.plot(list(data5.keys())[:11], list(data11.values())[:11], label='LSTM-JSCC $V_{test}=75m/s$', marker='o',ls='--',color='yellow')  # 使用方形标记数据点

ax.plot(list(data6.keys())[:11], list(data6.values())[:11], label='CNN-JSCC', marker='+' ,color='purple')  # 使用方形标记数据点   
# ax.plot(list(data6.keys())[:11], list(data12.values())[:11], label='CNN-JSCC $V_{test}=75m/s$', marker='+' ,ls='--',color='purple')  # 使用方形标记数据点   

ax.plot(list(data4.keys())[:11], list(data4.values())[:11], label='OTFS-JSCC',marker='^',color='orange')
# ax.plot(list(data4.keys())[:11], list(data10.values())[:11], label='OTFS-JSCC $V_{test}=75m/s$',marker='^',color='red',ls='--')

ax.plot(list(data4.keys())[:11], list(data14.values())[:11], label= 'BPG+1/2LDPC+QPSK', marker='.',color='grey')

# ax.plot(list(data7.keys())[:11], list(data7.values())[:11], label='jscc',marker='^',ls='--')

# 绘制第二组数据
# # ax.plot(list(data1.keys()), list(data1.values()), label='BPG 2/3LDPC QPSK', marker='.') ax.plot(list(data3.keys())[1:11], list(data3.values())[1:11], label='MA-PA-JSCC V=25m/s', marker='*',color='green')  # 使用方形标记数据点
# ax.plot(list(data3.keys())[:11], list(data3.values())[:11], label='MR-JSCC V=25m/s', marker='*',color='green')  # 使用方形标记数据点

# ax.plot(list(data6.keys())[:11], list(data6.values())[:11], label='MR-JSCC V=75m/s', marker='*',ls='--',color='green')  # 使用方形标记数据点

# ax.plot(list(data2.keys())[:11], list(data2.values())[:11], label='RES-JSCC V=25m/s', marker='v',color='blue')  # 使用方形标记数据点
# # print(list(data5.values()))
# ax.plot(list(data5.keys())[:11], list(data5.values())[:11], label='RES-JSCC V=75m/s', marker='v',ls='--',color='blue')  # 使用方形标记数据点
# # ax.plot(list(data4.keys())[:11],data5[:11], label='DL-CE-EQ-JSCC V=75m/s', marker='v',ls='--',color='blue')
# ax.plot(list(data4.keys())[:11], list(data4.values())[:11], label='OTFS-JSCC V=25m/s',marker='^',color='red')
# ax.plot(list(data7.keys())[:11], list(data7.values())[:11], label='OTFS-JSCC V=75m/s',marker='^',ls='--',color='red')
# 添加图例
# 添加图例
ax.legend()
plt.grid(True)
# 添加标题和轴标签
# ax.set_title('Train 6db, Test -4-16db')
ax.set_xlabel('SNR(dB) ')
ax.set_ylabel('SSIM')

# 显示图形
plt.show()
plt.savefig('cpv50SNRSSIM.png')
plt.savefig("cpv50SNRSSIM.eps", format="eps", dpi=600)
