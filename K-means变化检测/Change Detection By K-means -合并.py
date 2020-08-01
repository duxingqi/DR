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
                #log_img[i, j, k] = abs(rate_img[i, j, k])
                #print(log_img)                       
    return log_img


'''--------调用K-means算法对提取到的变化信息分类并显示图片-----------------------------------------'''

def kmeans(log_img):                      
    [rows, cols, chls] = log_img.shape
    log_img_one = log_img[:,:,0]
    pre_img = log_img_one.reshape((rows * cols, 1))
    #print(pre_img)
    k=2
    list_img = [] #初始聚类中心列表         
    while list_img.__len__() < k:
        n = np.random.randint(0,pre_img.shape[0],1)#以1为步长，随机选取0到行数间的数据        
        if n not in list_img:
            list_img.append(n[0])                  #选取初始聚类中心
    #print(list_img)
    pre_point = pre_img[np.array(list_img)]
    #print(list_img)
#①定义一个空列表
#②while循环，当列表里的个数等于给定的聚类中心时停止
#③从0到行数值之间以1为步长，随机选数字
#④如果不再列表中则加入
#⑤找到初始聚类点，跳出循环
#⑥至此，完成了初始聚类中心的选择
    c=0
    while True:
        distance = np.sqrt([np.sum((pre_img - i) ** 2, axis=1) for i in pre_point]) 
        now_point = np.argmin(distance, axis=0)       
        now_piont_distance = np.array(list([np.average(pre_img[now_point == i], axis=0) for i in range(k)]))        
        c+=0
        if np.sum(now_piont_distance - pre_point)<1e-5 or c>50:
            break           
        else:
            pre_point = now_piont_distance
#①欧氏距离的计算
#②将距离最小点选出
#③计算新选点距离，方便与前聚类中心比较
#④若聚类中心不再变化或者迭代次数大于50次跳出
#⑤差异信息的分类完成
    labels=now_point
    res = labels.reshape((rows, cols))
    ima = np.zeros([rows, cols, chls])          
    ima[:, :, 0] = res                          
    ima[:, :, 1] = res
    ima[:, :, 2] = res                          
    plt.title("K-means-Change Detection Results")       
    plt.imshow(ima)                                                                            
    plt.axis('off')                             
    plt.show()                                 

if __name__ == '__main__':
    img1 = mpimg.imread('1999.04.bmp')          
    img2 = mpimg.imread('1999.05.bmp')          
    dim = lograte(img1,img2)                    
    dim = (dim*255).astype(np.double)              
    kmeans(dim)



   
