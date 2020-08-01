# SAR Image change detection by k-means algorithm
#author:DR
#date:2020-07-08

'''-------------调用库-----------------------------------------------------------------------------'''

import matplotlib.pyplot as plt                 #图形库，plt 用于显示图片
import matplotlib.image as mpimg                #mpimg 用于读取图片
import numpy as np                              #科学运算库，处理数值计算和矩阵信息
import pandas as pd                             #基于numy数组构建，预处理数据，将1/2维数据转化为列表
import math                                     #实现数学函数运算
from sklearn.cluster import KMeans              #从科学计算库中调用Kmeans（各种聚类算法）

'''-----------定义两图片取对数函数----------------------------------------------------------------'''

def lograte(img1,img2):                         #对数比值函数
    [rows, cols, chls] = img1.shape             #把图片1像素的行数、列数以及通道数（维度）返回给rows，cols，chls
    rate_img = np.zeros([rows, cols, chls])     #返回一个用0填充的数组
    log_img = np.zeros([rows, cols, chls])      #(初始化数组)
    #print(log_img)
    for i in range(rows):                       #三特征循环遍历
        for j in range(cols):
            for k in range(chls):               #数组做除法返回浮点数而不做截断获得两个图像的比值,都加0.1使其为浮点数结果
                #print(img1[i,j,k])
                rate_img[i, j, k] = np.true_divide(img1[i, j, k]+0.1, img2[i, j, k]+0.1) # 2张图片的比值（一列长串数字作除）
                #print(img1[i, j, k])           #作除后的一长串浮点数值
                log_img[i, j, k] = abs(math.log(rate_img[i, j, k]))
                #print(log_img[i, j, k])        #取对数和绝对值后返回
                #print(log_img)
                
    return log_img

'''--------调用Kmeans算法对提取到的变化信息分类并显示图片------------------------------------------'''

def kmeans_chde(log_img):                       #采用kmeans进行变化检测
    [rows, cols, chls] = log_img.shape          #获得的对数矩阵信息返回给rows，cols，chls
    #print([rows, cols, chls])
    log_img_one = log_img[:,:,0]                #使得rgb通道的数值为0
    #print(log_img_one)
    I = log_img_one.reshape((rows * cols, 1))   #重新调整矩阵的行数、列数、维数，转化为1列数据后赋值给I
    #print(I)
    chde = KMeans(n_clusters = 2)               #聚类簇为2即分为2类
    chde.fit(I)                                 #数据预处理拟合，开始聚类
    labels = chde.labels_                       #每个数据点所属类别，这里是0和1    
    res = labels.reshape((rows, cols))          #将数据重新组织，恢复行列信息，将一列数据转化为图片大小
    #print(res)
    ima = np.zeros([rows, cols, chls])          #初始化数组
    ima[:, :, 0] = res                          #将数组信息恢复通道信息
    #print(ima[:, :, 0])
    ima[:, :, 1] = res
    #print(ima[:, :, 1])
    ima[:, :, 2] = res                          #恢复3通道
    #print(ima[:, :, 2])
    plt.title("Kmeans Change Detection Results")#添加标题
    plt.imshow(ima)                             #处理图像并显示格式
                                                #后面跟plt.show()才可
    plt.axis('off')                             #不显示坐标轴
    plt.show()                                  #显示图片
    
'''------定义主函数，调用并定义值，导入图片---------------------------------------------------------'''

if __name__ == '__main__':
    img1 = mpimg.imread('1999.04.bmp')          #读取图片1
    img2 = mpimg.imread('1999.05.bmp')          #读取图片2
    dim = lograte(img1,img2)                    #两图片比值对数信息赋值给dim
    dim = (dim*255).astype(np.int)              #转换数组的数据类型为整数型
    kmeans_chde(dim)                            #dim（变化信息）经Kmeans算法调用

    
