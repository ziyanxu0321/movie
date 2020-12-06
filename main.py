import pandas as pd
import os
import numpy as np
from csv import reader
from sklearn.cluster import KMeans
import joblib
from sklearn import cluster
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import train
import urllib
import requests
import urllib.request
from bs4 import BeautifulSoup

# 搜集数据
path = r"C:\Users\Administrator\Desktop\2020\毕设题目\数据\IMDb Top 250 - IMDb(others).csv"
file = open(path, 'r')
data = pd.read_csv(file)
# print(d)
# print(data.describe())
page0 = data.iloc[0:250, 15:16].values
page1 = []
for i in range(len(page0)):
    a = page0[i, 0]
    page1.append(a)
kewords = []
#print(type(page1))
# convert to string type
page = ['empty']*len(page1)
for i in range(len(page1)):
    page[i] = str(page1[i])

#print(len(page), type(page[1]))

def spider():
    data = 
for pg in page:
    print(pg)
    kw = urllib.request.urlopen(pg).read()
    soup = BeautifulSoup(pg, 'html.parser')

#path2 = r"C:\Users\Administrator\Desktop\2020\毕设题目\数据\IMDb Top 250 - IMDb(without text data).csv"
#ds = pd.read_csv(open(path2, 'r'))
#link = ds.iloc[0:250, 1:2].values
#soup = BeautifulSoup(link, 'lxml')



rankTitle = data.iloc[0:250, 0:1].values
# print(rankTitle)
rating = data.iloc[0:250, 1:2].values
# rating = rating.values
# print(rating)
ratingNo0 = data.iloc[0:250, 2:3].values
ratingNo = ratingNo0.astype(np.float)
# print(np.isnan(ratingNo).any())
df1 = pd.DataFrame(ratingNo)
df1.dropna(inplace=True)
ratingNo = np.array(df1)
# print(ratingNo.ndim, ratingNo.dtype, type(ratingNo))
# print(np.isnan(ratingNo).any())
# print(ratingNo[2][0])
# print(ratingNo)
reviews0 = data.iloc[0:250, 3:4].values
reviews = reviews0.astype(np.float)
# print(np.isnan(reviews).any())
df2 = pd.DataFrame(reviews)
df2.dropna(inplace=True)
reviews = np.array(df2)
# print(reviews.ndim, reviews.dtype, type(reviews))
# print(np.isnan(reviews).any())
# print(reviews[2][0])
# print(reviews)
director = data.iloc[0:250, 4:5].values
# print(director)
genres = data.iloc[0:250, 5:6].values
# print(genres)
country = data.iloc[0:250, 6:7].values
# print(country)
runtime0 = data.iloc[0:250, 11:12].values
df3 = pd.DataFrame(runtime0)
df3.dropna(inplace=True)
runtime = np.array(df3)
# print(runtime.dtype)
matrix = []
for i in range(0, 250):
    a = np.array(i)
    matrix.append(a)
# print(matrix)
estimator = KMeans(n_clusters=3)
res = estimator.fit_predict(rating)
lable_pred = estimator.labels_
centroids = estimator.cluster_centers_
inertia = estimator.inertia_
# print(lable_pred)
# print(centroids)
# print(inertia)


# print(type(rating0))
# print(len(rating0))
# print(rating0.ndim)
# print(rating0[1][-1])
for i in range(len(rating)):
    if int(lable_pred[i]) == 0:
        plt.scatter(matrix[i], rating[i][0], s=None, color='red')
    if int(lable_pred[i]) == 1:
        plt.scatter(matrix[i], rating[i][0], s=None, color='black')
    if int(lable_pred[i]) == 2:
        plt.scatter(matrix[i], rating[i][0], s=None, color='blue')
# plt.show()

estimatorNo = KMeans(n_clusters=5)
resNo = estimatorNo.fit_predict(ratingNo)
lable_predNo = estimatorNo.labels_
centroidsNo = estimatorNo.cluster_centers_
inertiaNo = estimatorNo.inertia_
# print(lable_predNo)
# print(centroidsNo)
# print(inertiaNo)

for i in range(len(ratingNo)):
    if int(lable_predNo[i]) == 0:
        plt.scatter(matrix[i], ratingNo[i][0], s=None, color='red')
    if int(lable_predNo[i]) == 1:
        plt.scatter(matrix[i], ratingNo[i][0], s=None, color='black')
    if int(lable_predNo[i]) == 2:
        plt.scatter(matrix[i], ratingNo[i][0], s=None, color='blue')
    if int(lable_predNo[i]) == 3:
        plt.scatter(matrix[i], ratingNo[i][0], s=None, color='yellow')
    if int(lable_predNo[i]) == 4:
        plt.scatter(matrix[i], ratingNo[i][0], s=None, color='green')
# plt.show()


estimatorRev = KMeans(n_clusters=6)
resRev = estimatorRev.fit_predict(reviews)
lable_predRev = estimatorRev.labels_
centroidsRev = estimatorRev.cluster_centers_
inertiaRev = estimatorRev.inertia_
# print(lable_predRev)
# print(centroidsRev)
# print(inertiaRev)

for i in range(len(reviews)):
    if int(lable_predRev[i]) == 0:
        plt.scatter(matrix[i], reviews[i][0], s=None, color='red')
    if int(lable_predRev[i]) == 1:
        plt.scatter(matrix[i], reviews[i][0], s=None, color='black')
    if int(lable_predRev[i]) == 2:
        plt.scatter(matrix[i], reviews[i][0], s=None, color='blue')
    if int(lable_predRev[i]) == 3:
        plt.scatter(matrix[i], reviews[i][0], s=None, color='yellow')
    if int(lable_predRev[i]) == 4:
        plt.scatter(matrix[i], reviews[i][0], s=None, color='green')
    if int(lable_predRev[i]) == 5:
        plt.scatter(matrix[i], reviews[i][0], s=None, color='purple')


# plt.show()

estimatorRun = KMeans(n_clusters=3)
resRun = estimatorRun.fit_predict(runtime)
lable_predRun = estimatorRun.labels_
centroidsRun = estimatorRun.cluster_centers_
inertiaRun = estimatorRun.inertia_
# print(lable_predRun)
# print(centroidsRun)
# print(inertiaRun)

for i in range(len(runtime)):
    if int(lable_predRun[i]) == 0:
        plt.scatter(matrix[i], runtime[i][0], s=None, color='red')
    if int(lable_predRun[i]) == 1:
        plt.scatter(matrix[i], runtime[i][0], s=None, color='black')
    if int(lable_predRun[i]) == 2:
        plt.scatter(matrix[i], runtime[i][0], s=None, color='blue')


# plt.show()

def distEclud(vecA,vecB):
    return np.sqrt(sum(np.power(vecA-vecB, 2)))

def randCent(dataSet,k):
    n = np.shape(dataSet)[1] # 计算列数

    centroids = np.mat(np.zeros((k,n)))
    for j in range(n):
        minJ = min(dataSet[:,j]) #取每列最小值
        rangeJ = float(max(dataSet[:,j])-minJ)
        centroids[:,j] = minJ + rangeJ*np.random.rand(k,1) # random.rand(k,1)构建k行一列，每行代表二维的质心坐标
        #random.rand(2,1)#产生两行一列0~1随机数
    return centroids

#minJ + rangeJ*random.rand(k,1)自动扩充阵进行匹配，实现不同维数矩阵相加,列需相同


#一切都是对象
def kMeans(dataSet,k,distMeas = distEclud,creatCent = randCent):
    m = np.shape(dataSet)[0] # 行数
    clusterAssment = np.mat(np.zeros((m, 2))) # 建立簇分配结果矩阵，第一列存索引，第二列存误差
    centroids = creatCent(dataSet, k) #聚类点
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf # 无穷大
            minIndex = -1 #初始化
            for j in range(k):
                distJI = distMeas(centroids[j,:],dataSet[i,:]) # 计算各点与新的聚类中心的距离
                if distJI < minDist: # 存储最小值，存储最小值所在位置
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        for cent in range(k):

            ptsInClust = dataSet[np.nonzero(clusterAssment[:,0].A== cent)[0]]
            # nonzeros(a==k)返回数组a中值不为k的元素的下标
            #print type(ptsInClust)
            '''
            #上式理解不了可见下面的，效果一样
            #方法二把同一类点抓出来

            ptsInClust=[]
            for j in range(m):
                if clusterAssment[j,0]==cent:
                    ptsInClust.append(dataSet[j].tolist()[0])
            ptsInClust = mat(ptsInClust)
            #tolist  http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tolist.html
            '''

            centroids[cent,:] = np.mean(ptsInClust,axis=0) # 沿矩阵列方向进行均值计算,重新计算质心
    return centroids, clusterAssment
# 构建二分k-均值聚类
def biKmeans(dataSet, k, distMeas=distEclud):
    m = np.shape(dataSet)[0]
    clusterAssment = np.mat(np.zeros((m,2))) # 初始化，簇点都为0
    centroid0 = np.mean(dataSet, axis=0).tolist()[0] # 起始第一个聚类点，即所有点的质心

    centList =[centroid0] # 质心存在一个列表中

    for j in range(m):#calc initial Error
        clusterAssment[j,1] = distMeas(np.mat(centroid0), dataSet[j,:])**2
        # 计算各点与簇的距离，均方误差，大家都为簇0的群

    while (len(centList) < k):

        lowestSSE = np.inf
        for i in range(len(centList)):

            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:,0].A==i)[0],:]#get the data points currently in cluster i
            # 找出归为一类簇的点的集合，之后再进行二分，在其中的簇的群下再划分簇
            #第一次循环时，i=0，相当于，一整个数据集都是属于0簇，取了全部的dataSet数据

            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            #开始正常的一次二分簇点
            #splitClustAss，类似于[0   2.3243]之类的，第一列是簇类，第二列是簇内点到簇点的误差

            sseSplit = sum(splitClustAss[:,1]) # 再分后的误差和
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:,0].A!=i)[0],1]) # 没分之前的误差
            print("sseSplit: ",sseSplit)
            print("sseNotSplit: ",sseNotSplit)
            #至于第一次运行为什么出现seeNoSplit=0的情况，因为nonzero(clusterAssment[:,0].A!=i)[0]不存在，第一次的时候都属于编号为0的簇

            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
                # copy用法http://www.cnblogs.com/BeginMan/p/3197649.html

        bestClustAss[np.nonzero(bestClustAss[:,0].A == 1)[0],0] = len(centList) #change 1 to 3,4, or whatever
        bestClustAss[np.nonzero(bestClustAss[:,0].A == 0)[0],0] = bestCentToSplit
        #至于nonzero(bestClustAss[:,0].A == 1)[0]其中的==1这簇点，由kMeans产生

        print('the bestCentToSplit is: ',bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))

        centList[bestCentToSplit] = bestNewCents[0,:].tolist()[0]#replace a centroid with two best centroids


        centList.append(bestNewCents[1,:].tolist()[0])

        clusterAssment[np.nonzero(clusterAssment[:,0].A == bestCentToSplit)[0],:]= bestClustAss#reassign new clusters, and SSE

    return np.mat(centList), clusterAssment


def show(dataSet, k, centriods, clusterA):
    import matplotlib.pyplot as plt
    numSamples, dim = np.shape(dataSet)
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    for i in range(numSamples):
        markIndex = int(clusterA[i, 0])
        plt.plot(dataSet[i, -1], dataSet[i, 0], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    for i in range(k):
        plt.plot(centriods[i, -1], centriods[i, 0], mark[i], markersize=12)
    plt.show()



estimator = KMeans(n_clusters=5)
res = estimator.fit_predict(ratingNo)
lable_pred = estimator.labels_
# print(lable_pred)
centroids = estimator.cluster_centers_
# print(centroids)
inertia1 = estimator.inertia_
# print(inertia1)

# myCentroids, clustAssing = kMeans(ratingNo, 5)
# show(ratingNo, 5, myCentroids, clustAssing)

# print(ratingNo.ndim, ratingNo.dtype, type(ratingNo))
# print(ratingNo)

ratingNo0 = data.iloc[0:250, 2:3].values
# ratingNo0[:, [0]] = matrix
#print(matrix.ndim)
ratingNo = ratingNo0.astype(np.float)
# print(np.isnan(ratingNo).any())
df1 = pd.DataFrame(ratingNo)
df1.dropna(inplace=True)
ratingNo = np.array(df1)
