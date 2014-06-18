'''
Created on Oct 24, 2013

@author: alecmacrae
'''
# project 2: knn

# establishes the arguments that should be used to run the script
import sys
import numpy

input_file_ALL = sys.argv[1]
input_file_AML = sys.argv[2]
k= sys.argv[3]
p = sys.argv[4]
output_file = "knn.out"

def transformIntoMatrix(data, leukemiaType):
    '''creates a matrix of the data and transposes it. 
    keeps track of leukemia leukemiaType as well by adding it to front of each row'''
    data = numpy.matrix(data)
    data = data.T
    
    #add leukemiaType to front of each row in matrix
    types = [leukemiaType]*len(data)
    types = numpy.matrix(types)
    types = types.T
    data = numpy.hstack([types,data])
    return data

def createFolds(ALL,AML):
    '''creates the 4 folds randomly that will later be used for cross validation'''
    numpy.random.shuffle(ALL)
    numpy.random.shuffle(AML)
    all1 = ALL[0:(len(ALL)/4.0)]
    all2 = ALL[(len(ALL)/4.0):(len(ALL)/2.0)]
    all3 = ALL[(len(ALL)/2.0):(len(ALL)*3.0/4.0)]
    all4 = ALL[(len(ALL)*3.0/4.0):len(ALL)]
    aml1 = AML[0:(len(AML)/4.0)]
    aml2 = AML[(len(AML)/4.0):(len(AML)/2.0)]
    aml3 = AML[(len(AML)/2.0):(len(AML)*3.0/4.0)]
    aml4 = AML[(len(AML)*3.0/4.0):len(AML)]
    group1 = numpy.vstack([all1,aml1])
    group2 = numpy.vstack([all2,aml2])
    group3 = numpy.vstack([all3,aml3])
    group4 = numpy.vstack([all4,aml4])
    folds = [group1,group2,group3,group4]
    return folds

def combineTrainingFolds(train):
    '''combines the three training groups/folds into one matrix'''
    return numpy.vstack([train[0],train[1],train[2]])

def getLeukemiaType(kNearestNeighbors, k, p):
    '''etimates the leukemia type based on the k nearest neighbors'''
    p = float(p)
    pos = 0
    neg = 0
    leukemiaType = 0
    for i in xrange(k):
        if kNearestNeighbors[1][i] == 1:
            pos+=1
        else:
            neg+=1
    if neg <= 0:
        neg = .000001
    percentPos = pos/neg
    if percentPos > p:
        leukemiaType = 1
    return leukemiaType

def findDistances(entry, training, k, p):
    '''finds the ten closest neighbors to a given point and returns the estimated leukemia type'''
    k = int(k)
    inf = 99999
    kNearestNeighbors = []
    kNearestNeighbors.append([])
    kNearestNeighbors.append([])
    for i in xrange(k):
        kNearestNeighbors[0].append(inf)
        kNearestNeighbors[1].append(0)
    
    for i in xrange(len(training)):
        a = entry[:,1:]
        b = training[i,1:]
        dist = numpy.linalg.norm(a-b)
        for j in xrange(k):
            if dist < kNearestNeighbors[0][j]:
                temp = kNearestNeighbors[0][j]
                kNearestNeighbors[0][j] = dist
                kNearestNeighbors[1][j] = training[i,0]
                dist = temp
    leukemiaType = getLeukemiaType(kNearestNeighbors,k,p)
    return leukemiaType

def findSSA(TP,FP,TN,FN):
    '''finds the sensitivity, specificity, and accuracy'''
    TP = float(TP)
    FP = float(FP)
    TN = float(TN)
    FN = float(FN)
    total = TP+FP+TN+FN
    sensitivity = round(TP/(TP+FN),2)
    specificity = round(TN/(TN+FP),2)
    accuracy = round((TP+TN)/total,2)
    results = [sensitivity,specificity,accuracy]
    return results

def getResults(training, test, k, p):
    '''finds the results (sensitivity, specificity, and accuracy) based on the estimated leukemia types for the test sets.'''
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for entry in test:
        leukemiaType = findDistances(entry,training,k,p) 
        if entry.item(0) == leukemiaType == 1:
            TP+=1
        elif entry.item(0) == leukemiaType == 0:
            TN+=1
        elif entry.item(0) != leukemiaType == 1:
            FP+=1
        elif entry.item(0) != leukemiaType == 0:
            FN+=1
    results = findSSA(TP,FP,TN,FN) # should return 3 things: sensitivity, specificity, and accuracy
    return results

def runTest(train, test, k, p):
    '''runs a cross validation test based on a given training set and test set'''
    training = combineTrainingFolds(train)
    results = getResults(training,test,k,p) # should return 3 things: sensitivity, specificity, and accuracy
    return results

def crossValidate(folds, k, p):
    '''cross validates a knn learning algorithm with 4 folds and returns resulting statistics'''
    resultsOne =   runTest((folds[1],folds[2],folds[3]), folds[0], k, p)
    resultsTwo =   runTest((folds[0],folds[2],folds[3]), folds[1], k, p)
    resultsThree = runTest((folds[0],folds[1],folds[3]), folds[2], k, p)
    resultsFour =  runTest((folds[0],folds[1],folds[2]), folds[3], k, p)
    sensitivity = round((resultsOne[0]+resultsTwo[0]+resultsThree[0]+resultsFour[0])/4.0,2)
    specificity = round((resultsOne[1]+resultsTwo[1]+resultsThree[1]+resultsFour[1])/4.0,2)
    accuracy =    round((resultsOne[2]+resultsTwo[2]+resultsThree[2]+resultsFour[2])/4.0,2)
    results = [sensitivity,specificity,accuracy]
    return results

allFile = open(input_file_ALL)
amlFile = open(input_file_AML)

ALL = allFile.readlines()
for i in xrange(len(ALL)):
    ALL[i] = ALL[i].strip().split()
    ALL[i] = map(float,ALL[i])
AML = amlFile.readlines()
for i in xrange(len(AML)):
    AML[i] = AML[i].strip().split()
    AML[i] = map(float,AML[i])

allFile.close()
amlFile.close()

ALL = transformIntoMatrix(ALL,1)
AML = transformIntoMatrix(AML,0)

folds = createFolds(ALL,AML) # folds is a vector of the 4 groups(vectors of vectors) of randomly selected data
results = crossValidate(folds,k,p) # returns a vector of three things: sensitivity, specificity, and accuracy

f = open(output_file, 'w')
f.write("k: " + str(k) + "\n")
f.write("p: " + str(p) + "\n")
f.write("accuracy: " + str(results[2]) + "\n")
f.write("sensitivity: " + str(results[0]) + "\n")
f.write("specificity: " + str(results[1]) + "\n")
f.close()
print ("k: " + str(k))
print ("p: " + str(p))
print ("accuracy: " + str(results[2]))
print ("sensitivity: " + str(results[0]))
print ("specificity: " + str(results[1]))

