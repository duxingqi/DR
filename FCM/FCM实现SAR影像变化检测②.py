import matplotlib.pyplot as plt              
import matplotlib.image as mpimg               
import matplotlib; matplotlib.use('TkAgg')
import numpy as np                              
import pandas as pd                            
import math                                   
import copy
import random

global MAX 
MAX = 10000.0
 
global gap 
gap = 0.0000001

def lograte(img1,img2):                         
    [rows, cols, chls] = img1.shape            
    rate_img = np.zeros([rows, cols, chls])    
    log_img = np.zeros([rows, cols, chls])      
    for i in range(rows):                       
        for j in range(cols):
            for k in range(chls):               
                rate_img[i, j, k] = np.true_divide(img1[i, j, k]+0.1, img2[i, j, k]+0.1) 
                log_img[i, j, k] = abs(rate_img[i, j, k])
      
    return log_img                            #得到图像变化部分                           

def initialize_U(data, cluster_number):       #构建隶属度矩阵
    
	"""

	这个函数是隶属度矩阵U的每行加起来都为1. 此处需要一个全局变量MAX.

	"""
	global MAX
	U = []
	#print(len(data))
	#print(data)
	for i in range(0, len(data)):  #遍历每一个数据点 (0,90601)     
		current = []
		rand_sum = 0.0
		for j in range(0, cluster_number):#遍历每一个聚类中心（cluster_number=2）
			index_cls = random.randint(1,int(MAX))#在1到MAX（10000.0）中随机选一个数
			current.append(index_cls)#把数添加到current列表中
			rand_sum += index_cls#选取的点赋值给rand_sum(完成了两个点的选取)
			#print(cluster_number)---->2*len(data)
		for j in range(0, cluster_number):
			current[j] = current[j] / rand_sum #第一第二个点取出与总和做差并将值添加到U中    
		U.append(current)
	#print(len(U))
	return U
        
 
def distance(point, center):  #计算点之间的距离（欧氏距离）                  
	if len(point) != len(center):  
		return -1#若点的个数不等于聚类中心的个数则返回-1
	sum_dis = 0.0
	sum_dis += abs(point - center) ** 2
	return math.sqrt(sum_dis)#计算点与聚类中心的欧氏距离
	"""

	该函数计算2点之间的距离（作为列表）。我们指欧几里德距离

	""" 
def end_conditon(U, U_old):  #隶属度矩阵迭代结束条件
    """
    结束条件。当U矩阵随着连续迭代停止变化时，触发结束
    
    """
    global gap
    for i in range(0, len(U)):
	    for j in range(0, len(U[0])):

		    if abs(U[i][j] - U_old[i][j]) > gap :#若新旧差值小于gap则报错，反之返回正确
			    return False
    return True
    


	

def chuli(U):                 
    ss = []
    for i in range(0, len(U)):#遍历隶属度矩阵每行
        maximum = max(U[i])#将每行最大隶属度值选出
        for j in range(0, len(U[0])):#遍历隶属度个数
            if U[i][j] == maximum:#如果某一隶属度为两个隶属度的较大值，则将其标记为最大隶属度
                ss.append(j) #将较大隶属度添加到ss列表
    U_new = np.array(ss)#将最大隶属度列表变为矩阵赋值给最新隶属度
    #print(np.array(ss))
    
    return U_new#返回
 
def fuzzy(data, cluster_number, m):  #主函数（数据，聚类数目，m=2）     
	
	U = initialize_U(data, cluster_number) #调用函数，计算隶属度U          
	while True:                                      	
		U_old = copy.deepcopy(U) #深层copy隶属度U（获取原始隶属度）     	
		C = []
		for j in range(0, cluster_number):#Ci的循环
			current_cluster_center = []
			for i in range(0, len(data[0])): #数据"行"的循环       
				sum_up = 0.0
				sum_do = 0.0
				for k in range(0, len(data)):   #数据"列"的循环        
					sum_up += (U[k][j] ** m) * data[k][i]   #分子
					sum_do += (U[k][j] ** m)                #分母
				current_cluster_center.append(sum_up/sum_do) 	#Ci	
			C.append(current_cluster_center) #获得的Ci的值添加到C列表中                          
		distance_matrix =[]
		for i in range(0, len(data)):#遍历数据行数
			current = []
			for j in range(0, cluster_number):#遍历聚类中心数
				current.append(distance(data[i], C[j]))#调用数据和簇中心距离，并加入current中
			distance_matrix.append(current)#将current添加到距离矩阵列表中
		
		for j in range(0, cluster_number):	
			for i in range(0, len(data)):
				sum_1 = 0.0
				for k in range(0, cluster_number):
					sum_1 += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2/(m-1))#公式的分母
				U[i][j] = 1 / sum_1#计算隶属度
		
		if end_conditon(U, U_old):#带入到判断结束函数中，判断是否结束
			break
	
	U = chuli(U)  #结束后将获取到的隶属度做归一化处理，选出最佳隶属度
	return U 

'''------------------------最佳隶属度U_newq确定------------------------------------------------------------'''
def FCM_chde(log_img):                       
    [rows, cols, chls] = log_img.shape          #获得的对数矩阵信息返回给rows，cols，chls
    log_img_one = log_img[:,:,0]                #获取所有二维数组的0列,后返回给log_img_one
    I = log_img_one.reshape((rows * cols, 1))   #重新调整矩阵的行数、列数、维数，转化为1列数据后赋值给I   reshape里括号表示几行几列
    labels = fuzzy(I,2,2)                       #归一化信息带入fuzzy函数，计算出最新隶属度赋值给labels
    res = labels.reshape((rows, cols))          #将数据重新组织，恢复行列信息，将一列数据转化为图片大小
    ima = np.zeros([rows, cols, chls])          #初始化数组
    ima[:, :, 0] = res                          #将数组信息恢复通道信息
    ima[:, :, 1] = res
    ima[:, :, 2] = res                          #恢复3通道
    plt.title("Change Detection Results")       #添加标题
    plt.imshow(ima)                             #处理图像并显示格式
    plt.axis('off')                             #不显示坐标轴
    plt.show()                                  #显示图片 

if __name__ == '__main__':
    img1 = mpimg.imread('1999.04.bmp')          #读取图片1
    img2 = mpimg.imread('1999.05.bmp')          #读取图片2
    dim = lograte(img1,img2)                    #两图片比值对数信息赋值给dim
    dim = (dim*255).astype(np.int)              #转换数组的数据类型为整数型
    FCM_chde(dim)                            

    
