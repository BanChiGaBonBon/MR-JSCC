import matplotlib.pyplot as plt
import numpy as np
d2 = 'L25.0316s16m30res'
data2 = np.load('%s.npy'%d2,allow_pickle=True).tolist()


d3 = 'L25.0316s16m30tcm'
data3 = np.load('%s.npy'%d3,allow_pickle=True).tolist()

d4 = 'L25.0316s16m30ceeqotfscm'
data4 = np.load('%s.npy'%d4,allow_pickle=True).tolist()

d5 = 'L75.0316s16m30res'
data5 = np.load('%s.npy'%d5,allow_pickle=True).tolist()

d6 = 'L75.0316s16m30tcm'
data6 = np.load('%s.npy'%d6,allow_pickle=True).tolist()

d7 = 'L75.0316s16m30ceeqotfscm'
data7 = np.load('%s.npy'%d7,allow_pickle=True).tolist()


data8 = np.load('L25.0405s16m30ccm.npy',allow_pickle=True).tolist()
data9 = np.load('L75.0405s16m30ccm.npy',allow_pickle=True).tolist()


data10 = np.load('L75.0406s16m30lstmcm.npy',allow_pickle=True).tolist()
data11 = np.load('L25.0406s16m30lstmcm.npy',allow_pickle=True).tolist()

time_points = range(0,110,10)


fig, ax = plt.subplots()


# 绘制第二组数据
# ax.plot(list(data1.keys()), list(data1.values()), label='BPG 2/3LDPC QPSK', marker='.') 
ax.plot(list(data3.keys())[1:11], list(data3.values())[1:11], label='MR-JSCC V=25m/s', marker='*',color='green')  # 使用方形标记数据点
ax.plot(list(data6.keys())[1:11], list(data6.values())[1:11], label='MR-JSCC V=75m/s', marker='*',ls='--',color='green')  # 使用方形标记数据点


ax.plot(list(data2.keys())[1:11], list(data2.values())[1:11], label='DL-CE-EQ-JSCC V=25m/s', marker='v',color='blue')  # 使用方形标记数据点
ax.plot(list(data5.keys())[1:11], list(data5.values())[1:11], label='DL-CE-EQ-JSCC V=60m/s', marker='v',ls='--',color='blue')  # 使用方形标记数据点

ax.plot(list(data4.keys())[1:11], list(data4.values())[1:11], label='NU-CE-EQ-JSCC V=25m/s',marker='^',color='red')
ax.plot(list(data7.keys())[1:11], list(data7.values())[1:11], label='NU-CE-EQ-JSCC V=75m/s',marker='^',ls='--',color='red')
ax.plot(list(data4.keys())[1:11], list(data8.values())[1:11], label='cnn-JSCC V=25m/s')
ax.plot(list(data4.keys())[1:11], list(data9.values())[1:11], label='cnn-JSCC V=75m/s')
ax.plot(list(data4.keys())[1:11], list(data10.values())[1:11], label='lstm-JSCC V=25m/s')
ax.plot(list(data4.keys())[1:11], list(data11.values())[1:11], label='lstm-JSCC V=25m/s')


# 添加图例
ax.legend()
plt.grid(True)
# 添加标题和轴标签
# ax.set_title('Train 6db, Test -4-16db')
ax.set_xlabel('Number of Path')
ax.set_ylabel('PSNR(dB)')

# 显示图形
plt.show()
plt.savefig('cpv50L.png')
plt.savefig("cpv50L.eps", format="eps", dpi=600)
