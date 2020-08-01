'''--------调用库--------------------------------------------------------------------------------'''
import matplotlib.pyplot as plt              
import matplotlib.image as mpimg               
import numpy as np                              
import pandas as pd                            
import math                                   
import copy
import random
'''---------全局变量------------------------------------------------------------------------------'''
global MAX 
MAX = 10000.0
global gap 
gap = 0.0000001
'''---------图像做差去对数函数--------------------------------------------------------------------'''
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
'''--------以可重复的方式打印矩阵----------------------------------------------------------------'''
def print_matrix(list):
	for i in range(0, len(list)):

		print (list[i])
'''--------这个函数是隶属度矩阵U的每行加起来都为1. 此处需要一个全局变量MAX.----------------------'''
def initialize_U(data, cluster_number):             
	global MAX
	U = []
	for i in range(0, len(data)):        
		current = []
		rand_sum = 0.0
		for j in range(0, cluster_number):
			index_cls = random.randint(1,int(MAX))
			current.append(index_cls)
			rand_sum += index_cls
		for j in range(0, cluster_number):
			current[j] = current[j] / rand_sum     
		U.append(current)
	return U
'''--------该函数计算2点之间的距离。指欧几里德距离----------------------------------------------'''
def distance(point, center):                    
	if len(point) != len(center):  
		return -1
	sum_dis = 0.0
	sum_dis += abs(point - center) ** 2
	return math.sqrt(sum_dis)
'''--------结束条件,当U矩阵随着连续迭代停止变化时，触发结束------------------------------------'''
def end_conditon(U, U_old):                    
    global gap
    for i in range(0, len(U)):
	    for j in range(0, len(U[0])):              
		    if abs(U[i][j] - U_old[i][j]) > gap :
			    return False
    return True
'''--------判断隶属度的更新--------------------------------------------------------------------''' 
def chuli(U):                 
    ss = []
    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] == maximum:
                ss.append(j)        
    U_new = np.array(ss)     
    return U_new
'''--------主函数将计算所需的聚类中心，并返回最终的归一化隶属矩阵U.--------------------------'''
def fuzzy(data, cluster_number, m):       	
	U = initialize_U(data, cluster_number)           
	while True:                                      	
		U_old = copy.deepcopy(U)      	
		C = []
		for j in range(0, cluster_number):
			current_cluster_center = []
			for i in range(0, len(data[0])):        
				sum_up = 0.0
				sum_do = 0.0
				for k in range(0, len(data)):           
					sum_up += (U[k][j] ** m) * data[k][i]    
					sum_do += (U[k][j] ** m)                 
				current_cluster_center.append(sum_up/sum_do) 		
			C.append(current_cluster_center)                           		
		distance_matrix =[]
		for i in range(0, len(data)):
			current = []
			for j in range(0, cluster_number):
				current.append(distance(data[i], C[j]))
			distance_matrix.append(current)		
		for j in range(0, cluster_number):	
			for i in range(0, len(data)):
				sum_1 = 0.0
				for k in range(0, cluster_number):
					sum_1 += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2/(m-1))      	#公式的分母
				U[i][j] = 1 / sum_1		
		if end_conditon(U, U_old):
			break	
	U = chuli(U)    
	return U 
'''-------输出函数，将差异图像分类后恢复三通道--------------------------------------------------------------'''
def FCM_chde(log_img):                       
    [rows, cols, chls] = log_img.shape          #获得的对数矩阵信息返回给rows，cols，chls
    log_img_one = log_img[:,:,0]                #获取所有二维数组的0列,后返回给log_img_one
    I = log_img_one.reshape((rows * cols, 1))   #重新调整矩阵的行数、列数、维数，转化为1列数据后赋值给I   reshape里括号表示几行几列
    labels = fuzzy(I,2,2)
    res = labels.reshape((rows, cols))          #将数据重新组织，恢复行列信息，将一列数据转化为图片大小
    ima = np.zeros([rows, cols, chls])          #初始化数组
    ima[:, :, 0] = res                          #将数组信息恢复通道信息
    ima[:, :, 1] = res
    ima[:, :, 2] = res                          #恢复3通道
    plt.title("Change Detection Results")       #添加标题
    plt.imshow(ima)                             #处理图像并显示格式
    plt.axis('off')                             #不显示坐标轴
    plt.show()                                  #显示图片 
'''-------执行函数------------------------------------------------------------------------------------------------'''
if __name__ == '__main__':
    img1 = mpimg.imread('1999.04.bmp')          #读取图片1
    img2 = mpimg.imread('1999.05.bmp')          #读取图片2
    dim = lograte(img1,img2)                    #两图片比值对数信息赋值给dim
    dim = (dim*255).astype(np.double)           #转换数组的数据类型转换
    FCM_chde(dim)                            

    
