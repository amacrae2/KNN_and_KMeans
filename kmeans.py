'''
Created on Oct 27, 2013

@author: alecmacrae
'''
# project 2: kmeans

# establishes the arguments that should be used to run the script
import sys
import numpy
from copy import deepcopy

def makeStartingCentroids(data, k):
    '''creates k centroids based on real data that is to be clustered'''
    centroids = []
    numpy.random.shuffle(data)
    for i in xrange(k):
        centroids.append(data[i])
    return centroids

def transformIntoMatrix(data):
    '''creates a matrix of the data'''
    data = numpy.matrix(data)
    return data

def labelArbitratyGroup(data):
    '''adds index and arbitrary cluster 1 to front of each row in matrix'''
    types = []
    types.append(range(1,len(data)+1))
    types.append([1]*len(data))
    types = numpy.matrix(types)
    types = types.T
    data = numpy.hstack([types,data])
    return data

def assignClustersToData(data, centroids, k):
    '''assigns a cluster number 1-k to each gene based on nearest centroid'''
    for i in xrange(len(data)):
        currClosestCentroid = 0
        currClosestDist = 99999
        for j in xrange(k):
            a = data[i,2:]
            b = centroids[j]
            dist = numpy.linalg.norm(a-b)
            if dist < currClosestDist:
                currClosestCentroid = j+1
                currClosestDist = dist
        data[i,1] = currClosestCentroid
    return data

def getNewCentroidLocations(data, centroids, k):
    '''finds the mean of all of the points in each cluster and relocates the respective centroids
    to this location'''
    data = numpy.squeeze(numpy.asarray(data))
    for i in xrange(k):
        group = data[data[:,1]==(i+1),2:]
        means = numpy.average(group,axis=0)
        centroids[i] = means
    return centroids

def performKMeans(data,centroids,k,maxIt):
    '''iterates through updates to the centroids locations until the centroids either stop moving,
    or the number of iterations reaches the maxt number'''
    iter = 1
    data = transformIntoMatrix(data)
    centroids = transformIntoMatrix(centroids)
    data = labelArbitratyGroup(data)
    while iter <= maxIt:
        temp = deepcopy(centroids)
        data = assignClustersToData(data,centroids,k)
        centroids = getNewCentroidLocations(data,centroids, k)
        iter+=1
        if numpy.allclose(centroids, temp):
            break
    return data

k = sys.argv[1]
maxIt = sys.argv[3]
k = int(k)
maxIt = int(maxIt)

input_file = sys.argv[2]
dataFile = open(input_file)
data = dataFile.readlines()
dataFile.close()
for i in xrange(len(data)):
    data[i] = data[i].strip().split()
    data[i] = map(float,data[i])

centroids = []
if len(sys.argv) > 4:
    startingCentroids_file = sys.argv[4]
    centroidFile = open(startingCentroids_file)
    centroids = centroidFile.readlines()
    centroidFile.close()
    for i in xrange(len(centroids)):
        centroids[i] = centroids[i].strip().split()
        centroids[i] = map(float,centroids[i])
else:
    dataCopy = deepcopy(data)
    centroids = makeStartingCentroids(dataCopy, k)
output_file = "kmeans.out"

data = performKMeans(data,centroids,k,maxIt)
f = open(output_file, 'w')
for i in xrange(len(data)):
    f.write(str(data.astype(int)[i,0]))
    f.write("\t")
    f.write(str(data.astype(int)[i,1]))
    f.write("\n")
f.close()
    