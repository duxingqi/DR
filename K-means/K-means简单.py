"""
    任务要求：对平面上的 100 个点进行聚类，要求聚类为两类，其横坐标都为 0 到 99。
"""
import numpy as np
import matplotlib.pyplot as plt
'''---------------0.导入库后初始化值-------------------------------------'''

x = np.linspace(0, 99, 100)
y = np.linspace(0, 99, 100)
k = 2
n = len(x)
dis = np.zeros([n, k+1])#初始化矩阵，100行3列
#print(dis)

#'''---------------1.选择初始聚类中心----------------------------------------------'''
center1 = np.array([x[0], y[0]])#定义初始化位置center1(0,0)
center2 = np.array([x[1], y[1]])#center2(1,1)
#print(center1)
iter_ = 100

while iter_ > 0:
#'''---------------2.求各个点到两个聚类中心距离------------------------------------'''

    for i in range(n):
        dis[i, 0] = np.sqrt((x[i] - center1[0])**2 + (y[i] - center1[1])**2)
        dis[i, 1] = np.sqrt((x[i] - center2[0])**2 + (y[i] - center2[1])**2)

#'''---------------3.归类------------------------------------------------------'''
        dis[i, 2] = np.argmin(dis[i,:2])  # 将值较小的下标值赋值给dis[i, 2]

#'''---------------4.求新的聚类中心------------------------------------------------'''
    index1 = dis[:, 2] == 0
    index2 = dis[:, 2] == 1
    center1_new = np.array([x[index1].mean(), y[index1].mean()])
    center2_new = np.array([x[index2].mean(), y[index2].mean()])

#'''---------------5.判定聚类中心是否发生变换-----------------------------------------'''
    if all((center1 == center1_new) & (center2 == center2_new)):
       # 如果没发生变换则退出循环，表示已得到最终的聚类中心
       break

    center1 = center1_new
    center2 = center2_new

'''---------------6.输出结果以验证------------------------------------------------'''
print(dis)







































