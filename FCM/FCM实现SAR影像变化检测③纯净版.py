import matplotlib.pyplot as plt              
import matplotlib.image as mpimg               
import numpy as np                              
import pandas as pd                            
import math                                   
import copy
import random
import time
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
def initialize_U(data, cluster_number):                                
	U = []
	for i in range(0, len(data)):                                    
		current = []
		rand_sum = 0.0
		for j in range(0, cluster_number):                     
			index_cls = random.randint(1,int(10000.0))     
			current.append(index_cls)                      
			rand_sum += index_cls                          
			                                                
		for j in range(0, cluster_number):
			current[j] = current[j] / rand_sum                  
		U.append(current)
	return U
def distance(point, center):                                          
	sum_dis = 0.0
	sum_dis += abs(point - center) ** 2
	return math.sqrt(sum_dis)
def end_conditon(U, U_old):                                           
    for i in range(0, len(U)):                                        
	    for j in range(0, len(U[0])):                              

		    if abs(U[i][j] - U_old[i][j]) > 1e-5 :              
			    return False                               
    return True
def chuli(U):                 
    max_U = []
    for i in range(0, len(U)):
        maximum = max(U[i])
        for j in range(0, len(U[0])):
            if U[i][j] == maximum:
                max_U.append(j) 
    U_new = np.array(max_U)   
    return U_new
def fuzzy(data, cluster_number, m):                                     
	U = initialize_U(data, cluster_number)                         
	while True:                                      	
		U_old = copy.deepcopy(U)                                 	
		Ci = []
		for j in range(0, cluster_number):                     
			current_cluster_center = []                    
			for i in range(0, len(data[0])):                    
				Molecular = 0.0
				Denominator = 0.0
				for k in range(0, len(data)):                
					Molecular += (U[k][j] ** m) * data[k][i]     
					Denominator += (U[k][j] ** m)                
				current_cluster_center.append(Molecular/Denominator) 	
			Ci.append(current_cluster_center)              	
		distance_matrix =[]
		for i in range(0, len(data)): 
			current = []                                    
			for j in range(0, cluster_number):              
				current.append(distance(data[i], Ci[j]))
			distance_matrix.append(current)                                                                 
		for j in range(0, cluster_number):	               
			for i in range(0, len(data)):                   
				sum_1 = 0.0
				for k in range(0, cluster_number):
					sum_1 += (distance_matrix[i][j] / distance_matrix[i][k]) ** (2/(m-1))
				U[i][j] = 1 / sum_1                     
		if end_conditon(U, U_old):                             
			break
	U = chuli(U)
	return U
def FCM_chde(log_img):
    [rows, cols, chls] = log_img.shape         
    log_img_one = log_img[:,:,0]                
    I = log_img_one.reshape((rows * cols, 1))   
    labels = fuzzy(I,2,2)                       
    res = labels.reshape((rows, cols))       
    ima = np.zeros([rows, cols, chls])          
    ima[:, :, 0] = res                         
    ima[:, :, 1] = res
    ima[:, :, 2] = res                          
    plt.title("FCM-Change Detection Results")       
    plt.imshow(ima)                            
    plt.axis('off')                            
    plt.show()                                  
if __name__ == '__main__':
    img1 = mpimg.imread('1999.04.bmp')          
    img2 = mpimg.imread('1999.05.bmp')          
    data = lograte(img1,img2)               
    data = (data*255).astype(np.double) 
    FCM_chde(data)                             

    
