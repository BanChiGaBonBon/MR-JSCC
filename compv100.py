import matplotlib.pyplot as plt
import numpy as np
# 示例数据
d1 = '1130s16m30gat100150'

data1 =dict(np.load('%s.npy'%d1,allow_pickle=True).tolist())


d2 = '1130s16m30resv100'
data2 = np.load('%s.npy'%d2,allow_pickle=True).tolist()
d3 = '1130s16m30resv110'
data3 = np.load('%s.npy'%d3,allow_pickle=True).tolist()
d4 = '1130s16m30resv120'
data4 = np.load('%s.npy'%d4,allow_pickle=True).tolist()
d5 = '1130s16m30resv130'
data5 = np.load('%s.npy'%d5,allow_pickle=True).tolist()
d6 = '1130s16m30resv140'
data6 = np.load('%s.npy'%d6,allow_pickle=True).tolist()
d7 = '1130s16m30resv150'
data7 = np.load('%s.npy'%d7,allow_pickle=True).tolist()

d8 = '1130s16m30ceeqotfsv100'
data8 = np.load('%s.npy'%d8,allow_pickle=True).tolist()
d9 = '1130s16m30ceeqotfsv110'
data9 = np.load('%s.npy'%d9,allow_pickle=True).tolist()
d10 = '1130s16m30ceeqotfsv120'
data10 = np.load('%s.npy'%d10,allow_pickle=True).tolist()
d11 = '1130s16m30ceeqotfsv130'
data11 = np.load('%s.npy'%d11,allow_pickle=True).tolist()
d12 = '1130s16m30ceeqotfsv140'
data12 = np.load('%s.npy'%d12,allow_pickle=True).tolist()
d13 = '1130s16m30ceeqotfsv150'
data13 = np.load('%s.npy'%d13,allow_pickle=True).tolist()

d14 = '1130s16m30resv100150'
data14 = np.load('%s.npy'%d14,allow_pickle=True).tolist()
d15 = '1130s16m30ceeqotfsv100150'
data15 = np.load('%s.npy'%d15,allow_pickle=True).tolist()
# 创建时间点（x轴）
time_points = range(0,110,10)

# 创建图形和轴
fig, ax = plt.subplots()

# 绘制第一组数据
#  # 使用圆圈标记数据点
ax.plot(list(data1.keys()), list(data1.values()), label='MR-JSCC', marker='*',color='green')

ax.plot(list(data2.keys()), list(data2.values()), label='RES-JSCC V=100m/s',marker='v',)
# ax.plot(list(data3.keys()), list(data3.values()), label='RES-JSCC V=110m/s' ,ls='--')
# ax.plot(list(data4.keys()), list(data4.values()), label='RES-JSCC V=120m/s' ,ls='--')
ax.plot(list(data5.keys()), list(data5.values()), label='RES-JSCC V=130m/s' ,marker='v',)
# ax.plot(list(data6.keys()), list(data6.values()), label='RES-JSCC V=140m/s' ,ls='--')
ax.plot(list(data7.keys()), list(data7.values()), label='RES-JSCC V=150m/s' ,marker='v',color='gray')
# ax.plot(list(data14.keys()), list(data14.values()), label='RES-JSCC V=100150m/s' ,ls='--')

ax.plot(list(data8.keys()), list(data8.values()), label='OTFS-JSCC V=100m/s' ,marker='^',)
# ax.plot(list(data9.keys()), list(data9.values()), label='OTFS-JSCC V=110m/s' )
# ax.plot(list(data10.keys()), list(data10.values()), label='OTFS-JSCC V=120m/s' ,marker='^',color='red')
ax.plot(list(data11.keys()), list(data11.values()), label='OTFS-JSCC V=130m/s' ,marker='^')
# ax.plot(list(data12.keys()), list(data12.values()), label='OTFS-JSCC V=140m/s' )
ax.plot(list(data12.keys()), list(data13.values()), label='OTFS-JSCC V=150m/s' ,marker='^')
# ax.plot(list(data12.keys()), list(data15.values()), label='OTFS-JSCC V=100150m/s' )
# 绘制第二组数据
# ax.plot(list(data3.keys())[:11], list(data3.values())[:11], label='MR-JSCC', marker='*',color='green')  # 使用方形标记数据点
# # ax.plot(list(data6.keys())[:11], list(data6.values())[:11], label='MA-PA-JSCC with OTFS', )  # 使用方形标记数据点   
# ax.plot(list(data1.keys()), list(data1.values()), label='BPG+1/2LDPC+QPSK+OFDM', marker='.') 
# ax.plot(list(data8.keys()), list(data8.values()), label='BPG+3/4LDPC+QPSK+OTFS', marker='.') 



# ax.plot(list(data2.keys())[:11], list(data2.values())[:11], label='RES-JSCC', marker='v',color='blue')  # 使用方形标记数据点
# # ax.plot(list(data5.keys())[:11], list(data5.values())[:11], label='JSCC DL-CE-EQ V=75m/s', marker='v',ls='--')  # 使用方形标记数据点

# ax.plot(list(data4.keys())[:11], list(data4.values())[:11], label='OTFS-JSCC',marker='^',color='red')
# ax.plot(list(data7.keys())[:11], list(data7.values())[:11], label='JSCC NU-CE-EQ V=75m/s',marker='^',ls='--')
# 添加图例

ax.legend()
plt.grid(True)
# 添加标题和轴标签
# ax.set_title('Train 0-50, Test 0-100')
ax.set_xlabel('Velocity(m/s)')
ax.set_ylabel('PSNR(dB)')

# 显示图形
plt.show()
plt.savefig('cpv50v100.png')
plt.savefig("cpv50v100.eps", format="eps", dpi=600)
