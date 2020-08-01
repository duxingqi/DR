# SAR Image change detection by k-means algorithm
#author:DR
#date:2020-07-08

'''-------------调用库-----------------------------------------------------------------------------'''

import matplotlib.pyplot as plt                 
import matplotlib.image as mpimg                
import numpy as np                              
import pandas as pd                        
import math                                    
         

'''-----------定义两图片取对数函数----------------------------------------------------------------'''

def lograte(img1,img2):                         
    [rows, cols, chls] = img1.shape
    rate_img = np.zeros([rows, cols, chls])     
    log_img = np.zeros([rows, cols, chls])      
    for i in range(rows):                     
        for j in range(cols):
            for k in range(chls):                                          
                rate_img[i, j, k] = np.true_divide(img1[i, j, k]+0.1, img2[i, j, k]+0.1)                          
                log_img[i, j, k] = abs(math.log(rate_img[i, j, k]))
                                       
    return log_img                              

'''--------调用K-means算法对提取到的变化信息分类并显示图片-----------------------------------------'''

def Kmeans_one(pre_img, point):#将数据信息分为K类
    list_img = []          #定义一个空列表 #存储k个0到图片，行数随机K个值的一维数据

    while list_img.__len__() < point:
        n = np.random.randint(0,pre_img.shape[0],1)#以1为步长，随机选取0到行数间的数据
        #print(img_data.shape[0])
        if n not in list_img:
            list_img.append(n[0])
            #print(index_cls)
#①定义一个空列表
#②while循环，当列表里的个数等于给定的聚类中心时停止
#③从0到行数值之间以1为步长，随机选数字
#④如果不再列表中则加入
#⑤找到初始聚类点，跳出循环
    pre_point = pre_img[np.array(list_img)]
    c=0
    while True:           
        distance = np.sqrt([np.sum((pre_img - i) ** 2, axis=1)for i in pre_point])  
        now_point = np.argmin(distance, axis=0)       
        now_piont_distance = np.array(list([np.average(pre_img[now_point == i], axis=0)for i in range(point)]))       
        diff = np.sum((now_piont_distance - pre_point) ** 2)
        c+=0 
        if diff < 1e-6 or c>50:            
            break       
        else:            
            pre_point = now_piont_distance  
    return now_point 

def kmeans_tow(log_img):                      
    [rows, cols, chls] = log_img.shape
    log_img_one = log_img[:,:,0]
    I = log_img_one.reshape((rows * cols, 1))#整理好的单列矩阵数据
    labels=Kmeans_one(I,2)#将处理好的数据分成两类（生产差异图像信息）
    res = labels.reshape((rows, cols))#恢复行列的矩阵
    ima = np.zeros([rows, cols, chls])          
    ima[:, :, 0] = res                          
    ima[:, :, 1] = res
    ima[:, :, 2] = res                          
    plt.title("Change Detection Results")       
    plt.imshow(ima)                                                                            
    plt.axis('off')                             
    plt.show()                                 
    
'''------定义主函数，调用并定义值，导入图片---------------------------------------------------------'''

if __name__ == '__main__':
    img1 = mpimg.imread('1999.04.bmp')          
    img2 = mpimg.imread('1999.05.bmp')          
    dim = lograte(img1,img2)                    
    dim = (dim*255).astype(np.double)              
    kmeans_tow(dim)                            
