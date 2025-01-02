import matplotlib.pyplot as plt
import numpy as np
# 示例数据
d1 = '0630s32sv0snr'

# data1 =dict(np.load('%s.npy'%d1,allow_pickle=True).tolist())
data1 = {0:0.6653441456864613,10:0.6649016646872551,20:0.3327415464425646}

d2 = '0724s32resv50SSIM'
data2 = np.load('%s.npy'%d2,allow_pickle=True).tolist()


d3 = '0720s32gatv50SSIM'
data3 = np.load('%s.npy'%d3,allow_pickle=True).tolist()

d4 = '0720s32ceeqv50SSIM'
data4 = np.load('%s.npy'%d4,allow_pickle=True).tolist()

d5 = '0720s32resv50SSIM'
data5 = np.load('%s.npy'%d5,allow_pickle=True).tolist()

d6 = '0720s32gatv50SSIM'
data6 = np.load('%s.npy'%d6,allow_pickle=True).tolist()

d7 = '0720s32ceeqv50SSIM'
data7 = np.load('%s.npy'%d7,allow_pickle=True).tolist()
# 创建时间点（x轴）
time_points = range(0,110,10)

# 创建图形和轴
fig, ax = plt.subplots()

# 绘制第一组数据
#  # 使用圆圈标记数据点

# 绘制第二组数据
ax.plot(list(data3.keys())[:11], list(data3.values())[:11], label='MA-PA-JSCC', marker='*',color='green')  # 使用方形标记数据点

ax.plot(list(data1.keys()), list(data1.values()), label='BPG+2/3LDPC+QPSK', marker='.') 
# ax.plot(list(data6.keys())[:11], list(data6.values())[:11], label='Proposed V=75m/s', marker='*',ls='--')  # 使用方形标记数据点


ax.plot(list(data2.keys())[:11], list(data2.values())[:11], label='DL-CE-EQ-JSCC', marker='v',color='blue')  # 使用方形标记数据点
# ax.plot(list(data5.keys())[:11], list(data5.values())[:11], label='JSCC DL-CE-EQ V=75m/s', marker='v',ls='--')  # 使用方形标记数据点

ax.plot(list(data4.keys())[:11], list(data4.values())[:11], label='NU-CE-EQ-JSCC',marker='^',color='red')
# ax.plot(list(data7.keys())[:11], list(data7.values())[:11], label='JSCC NU-CE-EQ V=75m/s',marker='^',ls='--')
# 添加图例
ax.legend()
plt.grid(True)
# 添加标题和轴标签
# ax.set_title('Train 0-50, Test 0-100')
ax.set_xlabel('Velocity(m/s)')
ax.set_ylabel('SSIM')

# 显示图形
plt.show()
plt.savefig('cpv50vSSIM.png')
