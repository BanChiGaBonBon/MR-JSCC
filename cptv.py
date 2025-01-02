import matplotlib.pyplot as plt
import numpy as np

data1 =dict(np.load('1126s16m30gat.npy',allow_pickle=True).tolist())
data2 =dict(np.load('1126s16m30gatn711.npy',allow_pickle=True).tolist())
fig, ax = plt.subplots()
ax.plot(list(data1.keys()), list(data1.values()), label='no-timevarying', marker='.') 
ax.plot(list(data2.keys()), list(data2.values()), label='timevarying', marker='.') 


ax.legend()
plt.grid(True)
# 添加标题和轴标签
# ax.set_title('Train 0-50, Test 0-100')
ax.set_xlabel('Velocity(m/s)')
ax.set_ylabel('PSNR(dB)')

# 显示图形
plt.show()
plt.savefig('tv.png')
# plt.savefig("cpv50v.eps", format="eps", dpi=600)
