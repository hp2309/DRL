import math
import random
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from npy_append_array import NpyAppendArray
from matplotlib import pyplot as plt

def plotMap(X):
    plt.imshow(X, cmap='gray')
    plt.xlabel('I',fontsize=20)
    plt.ylabel('J',fontsize=20)
    plt.show()

def makeMap():
    """returns np array of size 128x128 (room map)"""
    arr = np.zeros((64,64))
    arr[0,:] = np.ones((1,64))
    arr[63,:] = np.ones((1,64))
    arr[:,0] = np.ones((1,64))
    arr[:,63] = np.ones((1,64))

    for i in range(0,15):
        arr[i,25] = 1
    for i in range(0,15):
        arr[i,45] = 1
    for i in range(25,40):
        arr[i,25] = 1
    for i in range(33,48):
        arr[i,45] = 1
    for i in range(0,49):
        arr[i,15] = 1
    for i in range(25,63):
        arr[48,i] = 1
    for i in range(25,63):
        arr[25,i] = 1
    for i in range(25,36):
        arr[14,i] = 1
    for i in range(54,63):
        arr[14,i] = 1
    for i in range(4):
        for j in range(10):
            arr[i,j+25] = 1
    for i in range(4):
        for j in range(8):
            arr[i,j+55] = 1
    for i in range(7):
        for j in range(7):
            arr[i+25,j+25] = 1
    for i in range(7):
        for j in range(7):
            arr[i+25,j+56] = 1
    for i in range(6):
        for j in range(15):
            arr[j+30,i] = 1
    for i in range(6):
        for j in range(15):
            arr[j+49,i] = 1
    for i in range(3):
        for j in range(3):
            arr[j+40,i+35] = 1
    for i in range(8):
        for j in range(2):
            arr[j+40,i+50] = 1
    for i in range(6):
        for j in range(5):
            arr[j+5,i+5] = 1
    for i in range(8):
        for j in range(2):
            arr[j+55,i+32] = 1
    for i in range(2):
        for j in range(6):
            arr[j+55,i+48] = 1
    
    newarr = np.zeros((128,128))
    for i in range(128):
        for j in range(128):
            newarr[i,j] = arr[i//2, j//2]
    
    plt.imshow(newarr.T, cmap='gray')
    # plotMap(newarr.T)
    # np.savetxt("room.txt", newarr.T, fmt='%1.2f')
    # print(arr)

    plt.savefig("img.png")
    return newarr.T

def loadMap(fname):
    """show/return map store as txt file"""
    x = np.loadtxt(fname)
    # plt.imshow(x, cmap='gray')
    # plt.show()
    return x

def drawOccGrid(mapArray,orig=(65,63),ndim=(64,64),scanRadius=30):
    """draw/show/return occupancy grid for scan for a local map ndim, centered at orig, of scanradius"""
    localMap = np.zeros(ndim)
    occGrid = np.zeros(ndim)

    # plotMap(mapArray)

    for i in range(ndim[0]):
        for j in range(ndim[1]):
            x = orig[0] + i - ndim[0]//2
            y = orig[1] + j - ndim[1]//2
            if(x<1 or y <1 or x>mapArray.shape[0]-1 or y>mapArray.shape[1]-1):
                localMap[i,j] = 1
            else:
                localMap[i,j] = mapArray[x,y]

    # radar scan
    x0 = ndim[0]//2
    y0 = ndim[1]//2
    for theta in range(360):
        found_obs = 0
        for rad in range(1, scanRadius):
            if(found_obs):
                continue
            x = (int)(x0 + math.cos(math.radians(theta))*rad)
            y = (int)(y0 + math.sin(math.radians(theta))*rad)
            
            if(localMap[x,y] == 1):
                found_obs = 1
                occGrid[x,y] = 1
            else:
                occGrid[x,y] = 0.4
            # print(x,y)

    if True:
        localMap[x0,y0] = 0.6
        localMap[x0,y0+1] = 0.6
        localMap[x0+1,y0] = 0.6
        localMap[x0+1,y0+1] = 0.6
        occGrid[x0,y0] = 0.6
        occGrid[x0+1,y0] = 0.6
        occGrid[x0,y0+1] = 0.6
        occGrid[x0+1,y0+1] = 0.6

    maps = np.hstack((localMap, occGrid))

    # plotMap(maps)
    return occGrid

def checkNeigbours(occGrid, point):
    """checks nearest 8 neighbours for unknown gridpoint"""
    x = point[0]
    y = point[1]
    for i in range(-1,2):
        for j in range(-1,2):
            if(occGrid[x+i,y+j] == 0):
                return 1
    return 0

def findFrontiers(occGrid):
    """Finds possible frontiers for a given occupancy."""
    frontiers = []
    occ = occGrid.copy()
    for i in range(occGrid.shape[0]):
        for j in range(occGrid.shape[1]):
            if (occGrid[i,j] == 0.4 and checkNeigbours(occGrid, (i,j))):
                occ[i,j] = 0.2
                frontiers.append([i,j])

    return occ, np.asarray(frontiers)

def cluster(Fr):
    """cluster the frontiers into k groups"""
    k = Fr.shape[0]//10
    z = linkage(Fr, "ward")
    clusters = fcluster(z, k, criterion="maxclust")
    
    # plt.figure(figsize=(10,8))
    # plt.scatter(Fr[:,0], Fr[:,1] ,  c = clusters, cmap = "brg")
    # print(clusters)
    # plt.show()

    return clusters, k

def centroidFr(Fr, clusters, k):
    """return centroid of clusters"""
    clustercount = np.zeros(k)
    clustersumx = np.zeros(k)
    clustersumy = np.zeros(k)
    for i in range(Fr.shape[0]):
        clustersumx[clusters[i]-1] += Fr[i, 0]
        clustersumy[clusters[i]-1] += Fr[i, 1]
        clustercount[clusters[i]-1] += 1
    clustersumx = clustersumx//clustercount
    clustersumy = clustersumy//clustercount
    centFr = np.vstack((clustersumx,clustersumy))
    return centFr

def debug():
    """print hello"""
    print("hello")

if __name__ == "__main__":
    map = makeMap()
    # occ = np.zeros((1,1))
    # map = loadMap("room.txt")
    # iters = 1000
    # iterations = iters
    # with NpyAppendArray("data.npy") as f:
    #     while iters>0:
    #         while True:
    #             originx = random.randint(0,127)
    #             originy = random.randint(0,127)
    #             orig = (originx,originy)
    #             if(map[orig[0],orig[1]] == 0):
    #                 occ = drawOccGrid(map, orig=orig)
    #                 break
    #             # print(">> bad origin: trying again")
    #         (occFr, Fr) = findFrontiers(occ)
            
    #         # plotMap(occFr)

    #         clusters, k = cluster(Fr)
    #         centFr = centroidFr(Fr, clusters, k)

    #         # plt.plot(Fr[:,1], Fr[:,0], 'bo')    # Plot all frontiers
    #         # plt.plot(centFr[1,:], centFr[0,:], 'ro')    # Plot centroids of clusters
    #         # plt.imshow(occ, cmap='gray')
    #         # plt.show()
    #         # print(centFr.shape)

    #         # pos,occ,centfr
    #         pos = np.zeros((64,64))
    #         orientation = random.randint(0, 360)
    #         for rad in range(1,9):
    #             x = int(rad*math.cos(math.radians(orientation)))
    #             y = int(rad*math.sin(math.radians(orientation)))
    #             pos[32+y,32+x] = 0.5
    #         pos[32,32] = 1

    #         frontiers = np.zeros((64,64))
    #         for i in range(centFr.shape[1]):
    #             frontiers[int(centFr[0,i]), int(centFr[1,i])] = 1

    #         data = np.array([pos,occ,frontiers])
    #         # plt.imsave("batch1\\img{}.png".format(iterations-iters),np.hstack((data[0], data[1], data[2])))
    #         data = np.reshape(data, (1,3,64,64))    
    #         f.append(data)
    #         print(iterations-iters)
    #         iters-=1

    # print("done")

    # data = np.load("data.npy", mmap_mode='r')
    # print(data.shape)
    
    # np.savetxt("data.txt", data)

    
    



