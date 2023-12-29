import numpy as np

# # 从txt文件中加载数据
# data = np.loadtxt('anchors.txt')
# print(data.shape)
#
# # 将数据保存为npy文件
# np.save('anchors.npy', data)
#
# 从npy文件中加载数据
loaded_data = np.load('anchors.npy')
# 打印加载的数据
print(loaded_data.shape)