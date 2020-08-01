import numpy as np#引入numpy库一遍进行数值计算并取别名np，可处理矩阵数组等问题
import matplotlib.pyplot as plt#引入matplotlin库并取别名plt,可进行函数坐标图像绘制


# 定义距离函数来计算两点距离并返回计算结果
def distance(e1, e2):
    return np.sqrt((e1[0]-e2[0])**2+(e1[1]-e2[1])**2)#坐标间距离的计算公式（欧氏距离）

## 确定集合中心
def means(arr):
    return np.array([np.mean([e[0] for e in arr]), np.mean([e[1] for e in arr])])#坐标到聚类中心的距离

# arr中距离a最远的元素，用于初始化聚类中心
def farthest(k_arr, arr):#距离函数
    f = [0, 0]
    max_d = 0#初始化最大距离为0
    for e in arr:#循环由近（等于0时）到最远距离
        d = 0
        for i in range(k_arr.__len__()):#获取自定义类长度
            d = d + np.sqrt(distance(k_arr[i], e))#计算距离d为到每个点的距离
        if d > max_d:#通过循环比较确定最大距离并返回最大距离
            max_d = d
            f = e#最大距离
    return f#返回最大距离f值

# arr中距离a最近的元素，用于聚类
def closest(a, arr):
    c = arr[1]
    min_d = distance(a, arr[1])#最小距离表示
    arr = arr[1:]
    for e in arr:
        d = distance(a, e)#点之间距离
        if d < min_d:#循环比较距离，确定最小距离
            min_d = d
            c = e
    return c#返回最小距离值c


if __name__=="__main__":#程序入口，当程序运行时，下代码开始运行（但若以文件库的方式运行则不运行）
    ##生成二维随机坐标（没有找到数据集）size统计矩阵元素个数，或矩阵某一维上的元素个数
    arr = np.random.randint(100, size=(1000, 1, 2))[:, 0, :]#确定范围生成随机坐标1000个坐标

    #初始化聚类中心和聚类容器
    m = 5#确定5个聚类中心
    r = np.random.randint(arr.__len__() - 1)#随机点与聚类中心距离
    k_arr = np.array([arr[r]])
    cla_arr = [[]]
    for i in range(m-1):
        k = farthest(k_arr, arr)
        k_arr = np.concatenate([k_arr, np.array([k])])#把多个字符文本或数值连接在一起,实现合并的功能
        cla_arr.append([])

    # 迭代聚类
    n = 50#确定迭代次数
    cla_temp = cla_arr
    for i in range(n):    #迭代n次
        for e in arr:    #把集合里每一个元素聚到最近的类
            ki = 0        #假定距离第一个中心最近
            min_d = distance(e, k_arr[ki])#挨个与聚类中心比较计算出最小距离
            for j in range(1, k_arr.__len__()):#循环比较距离长度
                if distance(e, k_arr[j]) < min_d:#循环比较找到更近的聚类中心
                    min_d = distance(e, k_arr[j])#最小距离
                    ki = j
            cla_temp[ki].append(e)#列表尾部追bai加元素
        # 迭代更新聚类中心
        for k in range(k_arr.__len__()):
            if n - 1 == i:#迭代次数达到50次停止迭代跳出
                break
            k_arr[k] = means(cla_temp[k])
            cla_temp[k] = []

    #可视化展示
    col= ['pink', 'green', 'blue', 'yellow', 'red']#定义五种颜色粉，绿，蓝，黄，红
    for i in range(m):#五种颜色赋值给五个聚类中心
        plt.scatter(k_arr[i][0], k_arr[i][1], linewidth=10, color=col[i])#绘制散点图，包含函数图像宽度，颜色信息
        plt.scatter([e[0] for e in cla_temp[i]], [e[1] for e in cla_temp[i]], color=col[i])
    plt.show()#可视化展示
