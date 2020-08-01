'''-------------------导入库-----------------------------------------------------------------------'''
import matplotlib.pyplot as plt              
import matplotlib.image as mpimg               
import numpy as np                              
import pandas as pd                            
import math                                   
import copy
import random
import time
'''------------------提取影像差异并取对数------------------------------------------------------------'''
star=time.perf_counter()
def lograte(img1,img2):                         
    [rows, cols, chls] = img1.shape            
    rate_img = np.zeros([rows, cols, chls])    
    log_img = np.zeros([rows, cols, chls])      
    for i in range(rows):                       
        for j in range(cols):
            for k in range(chls):               
                rate_img[i, j, k] = np.true_divide(img1[i, j, k]+0.1, img2[i, j, k]+0.1) 
                log_img[i, j, k] = abs(rate_img[i, j, k])
    return log_img
end=time.perf_counter()
print("lograte:{0:>30}s".format(end-star))
#得到图像变化部分
#①完成了差异图像的提取，即将图像做除得到差异图像log_img--->data
'''-----------------------------初始化隶属度矩阵---------------------------------------------------'''
star=time.perf_counter()
def initialize_U(data, cluster_number):                                 #初始化隶属度矩阵
	U = []
	for i in range(0, len(data)):                                   #遍历每一个像素 (0-->90601) 301*301像素    
		current = []
		rand_sum = 0.0
		for j in range(0, cluster_number):                      #循环聚类个数 个 次数（cluster_number=2）
			index_cls = random.randint(1,int(10000.0))      #在1到MAX（10000.0）中随机选一个数（一共选2个）
			current.append(index_cls)                       #把数添加到current列表中
			rand_sum += index_cls                           #选取的点赋值给rand_sum(完成了两个点的选取)
			                                                #print(cluster_number)---->2*len(data)
		for j in range(0, cluster_number):
			current[j] = current[j] / rand_sum              #第一第二个点取出与总和做差并将值添加到U中    
		U.append(current)
	return U
end=time.perf_counter()
print("initialize_U:{0:>25}s".format(end-star))
#循环90601次，每次选取聚类簇数个随机值
#选取完的值放入列表current中
#将每次选到的两个随机点相加赋值给rand_sum
#将选取到的随机点（current中）与两数总和做除再赋值回current
#将处理完的随机数添加带隶属度列表U中
#②完成了隶属度的初始化U
'''---------------------------计算欧氏距离-------------------------------------------------------------------------'''
star=time.perf_counter()
def distance(point, center):                                            #计算点之间的距离（欧氏距离）
	sum_dis = 0.0
	sum_dis += abs(point - center) ** 2
	return math.sqrt(sum_dis)
end=time.perf_counter()
print("distance:{0:>28}s".format(end-star))
#计算点与聚类中心的欧氏距离
#计算数据点与聚类中心点的欧氏距离
#③完成数据与聚类中心欧氏距离的计算
'''----------------------------设定隶属度迭代停止条件--------------------------------------------------------------'''
star=time.perf_counter()
def end_conditon(U, U_old):                                             #隶属度矩阵迭代结束条件
    for i in range(0, len(U)):                                          #n*2阶矩阵“行数”
	    for j in range(0, len(U[0])):                               #n*2阶矩阵0行元素数

		    if abs(U[i][j] - U_old[i][j]) > 1e-5 :              #若新旧差值小于gap则报错，反之返回正确
			    return False                                #表单不提交
    return True
end=time.perf_counter()
print("end_conditon:{0:>25}s".format(end-star))
#提交表单
#遍历隶属度矩阵的每一行（i）的每一个（j）元素
#若该位置的值与上次变化不大（<1e-5）,就返回真值（程序可以往下运行）
#否则返回假值，程序继续迭代
#④判断迭代是否应该结束
'''--------------------------------选出最新隶属度-----------------------------------------------------------------'''
star=time.perf_counter()
def chuli(U):                 
    max_U = []
    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] == maximum:
                max_U.append(j) 
    U_new = np.array(max_U)   
    return U_new
end=time.perf_counter()
print("chuli:{0:>32}s".format(end-star))
#循环次数为矩阵行数
#将每行最大隶属度值选出赋值给maximum
#遍历每一行元素
#如果某一隶属度为两个隶属度的较大值，则将其标记为最大隶属度
#将最大隶属度添加到max_U列表
#将选出的最大隶属度数字组变为矩阵（一阶）赋值给最新隶属度
#⑤在隶属度矩阵中选出最大隶属度值赋值给U_new
'''----------------------------------完成隶属度和聚类中心的迭代-------------------------------------------------------'''
star=time.perf_counter()
def fuzzy(data, cluster_number, m):                                     #主要迭代函数（数据，聚类数目，m=2）
	U = initialize_U(data, cluster_number)                          #调用initialize_U函数，计算隶属度U
	while True:                                      	
		U_old = copy.deepcopy(U)                                #深层copy隶属度U（获取原始隶属度）赋值为U_old     	
		Ci = []
		for j in range(0, cluster_number):                      #循环次数为聚类个数次，Ci的循环
			current_cluster_center = []                     #建立当前聚类中心列表
			for i in range(0, len(data[0])):                #根据数据"行"的元素数进行循环       
				Molecular = 0.0
				Denominator = 0.0
				for k in range(0, len(data)):           #数据"列"次数的循环        
					Molecular += (U[k][j] ** m) * data[k][i]     #分子
					Denominator += (U[k][j] ** m)                #分母
				current_cluster_center.append(Molecular/Denominator) #Ci	
			Ci.append(current_cluster_center)               #获得的Ci的值添加到C列表中,
			'''完成Ci的计算'''	
		distance_matrix =[]                                     #定义矩阵距离列表
		for i in range(0, len(data)):                           #遍历次数为数据“行”数
			current = []                                    #定义当前值列表
			for j in range(0, cluster_number):              #遍历次数为聚类中心数
				current.append(distance(data[i], Ci[j]))#调用求欧氏距离函数，求数据点和簇中心距离
				                                        #并加入current列表中(一个数据点可以求的两个距离)
			distance_matrix.append(current)                 #将current添加到距离矩阵列表中
			                                                #添加后列表中为每个点的两个欧氏距离组成的矩阵
		for j in range(0, cluster_number):	                #j为聚类中心个数
			for i in range(0, len(data)):                   #i为数据点个数
				sum_1 = 0.0
				for k in range(0, cluster_number):
					sum_1 += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2/(m-1))
				U[i][j] = 1 / sum_1                     #计算隶属度Uij
		if end_conditon(U, U_old):                              #带入到判断结束函数中，判断是否结束
			break
	U = chuli(U)                                                    #结束后将获取到的隶属度做归一化处理，选出最佳隶属度
	return U
end=time.perf_counter()
print("fuzzy:{0:>31}s".format(end-star))
#⑥输出最佳隶属度U_new

'''------------------------最佳隶属度U_newq确定------------------------------------------------------------'''
'''-------------------------------主函数---------------------------------------------------------------------'''
def FCM_chde(log_img):
    star=time.perf_counter()
    [rows, cols, chls] = log_img.shape          #获得的对数矩阵信息返回给rows，cols，chls
    log_img_one = log_img[:,:,0]                #获取所有二维数组的0列,后返回给log_img_one
    I = log_img_one.reshape((rows * cols, 1))   #重新调整矩阵的行数、列数、维数，转化为1列数据后赋值给I,reshape里括号表示几行几列
    labels = fuzzy(I,2,2)                       #归一化信息带入fuzzy函数，计算出最新隶属度U_new赋值给labels
    res = labels.reshape((rows, cols))          #将数据重新组织，恢复行列信息，将一列数据转化为图片大小
    ima = np.zeros([rows, cols, chls])          #初始化数组
    ima[:, :, 0] = res                          #将数组信息恢复通道信息
    ima[:, :, 1] = res
    ima[:, :, 2] = res                          #恢复3通道
    plt.title("FCM-Change Detection Results")       #添加标题
    plt.imshow(ima)                             #处理图像并显示格式
    plt.axis('off')                             #不显示坐标轴
    plt.show()                                  #显示图片
end=time.perf_counter()
print("FCM_chde:{0:>29}s".format(end-star))
if __name__ == '__main__':

    img1 = mpimg.imread('1999.04.bmp')          #读取图片1
    img2 = mpimg.imread('1999.05.bmp')          #读取图片2
    data = lograte(img1,img2)                   #两图片比值对数信息赋值给dim
    data = (data*255).astype(np.double)         #转换数组的数据类型为浮点型
    star=time.perf_counter()
    FCM_chde(data)                              #将数据代入函数实现输出全过程
end=time.perf_counter()
print("main:{0:>29}s".format(end-star))
    
