import matplotlib.pyplot as plt
import numpy as np
# 示例数据
d1 = '0630s32sv0snr'

# data1 =dict(np.load('%s.npy'%d1,allow_pickle=True).tolist())
data1 = {0:22.1941531878219,10:22.176244622192957,20:21.88828535775958,30:21.59212399287,40:21.63392713352204,50:21.857446440967274,
         60:22.06542750357,70:22.01798307,80:21.96499995301,90:21.711873873,100:21.73869764355}

d2 = '316s16m30res'

data2 = np.load('%s.npy'%d2,allow_pickle=True).tolist()


d3 = '316s16m30tcm'
# d3 = '0724s32resv50'
data3 = np.load('%s.npy'%d3,allow_pickle=True).tolist()

d4 = '316s16m30ceeqotfscm'
data4 = np.load('%s.npy'%d4,allow_pickle=True).tolist()

d5 = '406s16m30lstmcm'
data5 = np.load('%s.npy'%d5,allow_pickle=True).tolist()

d6 = '405s16m30ccm'
data6 = np.load('%s.npy'%d6,allow_pickle=True).tolist()

d7 = '316s16m30cm'
data7 = np.load('%s.npy'%d7,allow_pickle=True).tolist()

data8 = {0:22.78143817851458,10:19.309975330510316,20:19.047718376515657,30:17.128394917119284,40:15.261138238712054}
data8 = {0:23.862, 10:23.839, 20:23.853,30: 23.827,40: 23.859,50:23.820}
# 创建时间点（x轴）
time_points = range(0,110,10)

# 创建图形和轴
fig, ax = plt.subplots()

# 绘制第一组数据
#  # 使用圆圈标记数据点

# 绘制第二组数据
ax.plot(list(data3.keys())[:11], list(data3.values())[:11], label='MR-JSCC', marker='*',color='green')  # 使用方形标记数据点
# ax.plot(list(data8.keys()), list(data8.values()), label='BPG+3/4LDPC+QPSK+OTFS', marker='.') 



ax.plot(list(data2.keys())[:11], list(data2.values())[:11], label='RES-JSCC', marker='v',color='blue')  # 使用方形标记数据点
ax.plot(list(data5.keys())[:11], list(data5.values())[:11], label='LSTM-JSCC', marker='o',color='yellow')  # 使用方形标记数据点
ax.plot(list(data6.keys())[:11], list(data6.values())[:11], label='CNN-JSCC', marker='+',color='purple')  # 使用方形标记数据点   

ax.plot(list(data4.keys())[:11], list(data4.values())[:11], label='OTFS-JSCC',marker='^',color='orange')
ax.plot(list(data1.keys()), list(data1.values()), label='BPG+1/2LDPC+QPSK', marker='.',color='grey') 

# ax.plot(list(data7.keys())[:11], list(data7.values())[:11], label='jscc',marker='^',ls='--')
# 添加图例

ax.legend()
plt.grid(True)
# 添加标题和轴标签
# ax.set_title('Train 0-50, Test 0-100')
ax.set_xlabel('Velocity(m/s)')
ax.set_ylabel('PSNR(dB)')

# 显示图形
plt.show()
plt.savefig('cpv50v.png')
plt.savefig("cpv50v.eps", format="eps", dpi=600)
