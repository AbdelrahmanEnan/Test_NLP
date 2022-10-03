from cgi import test
import math
def createVector():
    r = int(input("Enter the number of rows (number of vectors): "))
    c = int(input("Enter the number of columns (dimension of the vector): "))
    Vector = []
    for i in range(r):
        a = []
        for j in range(c):
            a.append(float(input()))
        Vector.append(a)
    return Vector

def createZeroVector(r,c):
    Vector = []
    for i in range(r):
        a = []
        for j in range(c):
            a.append(0)
        Vector.append(a)
    return Vector

def printVector(V):
    for i in range(len(V)):
        for j in range(len(V[0])):
            print(V[i][j], end=" ")
        print()

def distanceTwoVectors(V1,V2):
    if (len(V1)!=len(V2)):
        return("Vectors are not from the same dimension")
    beforeSqrt = 0
    for i in range(len(V1)):
        beforeSqrt += (V1[i]-V2[i])**2
    distance = math.sqrt(beforeSqrt)
    return distance

def distanceUpdate(Vector, clusterCenters, distances):
    for i in range(len(distances)):
        for j in range(len(distances[0])):
            distances[i][j] = distanceTwoVectors(Vector[i],clusterCenters[j])

def closest (distances):
    cls = 999999
    indx = 0
    for i in range(len(distances)):
        if (distances[i] < cls):
            cls = distances[i]
            indx = i
    return (indx+1)

def clusterUpdate(distances, cluster):
    change = 0
    for i in range(len(distances)):
        temp = closest(distances[i])
        if (temp != cluster[i]):
            cluster[i] = temp
            change += 1
    return change

def centerUpdate(Vectors, clusterCenters, cluster):
    for i in range(len(Vectors[0])):

        ### The following commented code is general for different number of clusters (k)  #

        #sum = [0.0,0.0,0.0] #sum = createZeroVector(1,len(clusterCenters))
        #numNodes = [0,0,0]  #numNodes = createZeroVector(1,len(clusterCenters))
        sum1 = 0.0
        sum2 = 0.0
        sum3 = 0.0
        numNodes1 = 0
        numNodes2 = 0
        numNodes3 = 0
        for j in range(len(Vectors)):
            #for k in range(len(sum)):
            #   if (Vectors[j][0] == k + 1):
            #       sum[k] += sum1+=Vectors[j][i]
            #       numNodes[k] += 1
            #   if (numNodes[k] == 0):
            #       numNodes[k] = 1
            #   clusterCenters[k][i] = sum[k]/numNodes[k]
            if (cluster[j] == 1):
                sum1+=Vectors[j][i]
                numNodes1 += 1
            elif (cluster[j] == 2):
                sum2+=Vectors[j][i]
                numNodes2 += 1
            elif (cluster[j] == 3):
                sum3+=Vectors[j][i]
                numNodes3 += 1
        if (numNodes1 == 0):
            numNodes1 = 1
        if (numNodes2 == 0):
            numNodes2 = 1
        if (numNodes3 == 0):
            numNodes3 = 1
        clusterCenters[0][i] = sum1/numNodes1
        clusterCenters[1][i] = sum2/numNodes2
        clusterCenters[2][i] = sum3/numNodes3

def clustring(Vectors, clusterCenters):
    distances = createZeroVector(len(Vectors),len(clusterCenters))
    # c,r = 3,12
    #distances = [[0 for x in range(c)] for y in range(r)] 
    #cluster = createZeroVector(12,1)
    cluster = [0 for x in range(len(Vectors))] 
    for i in range(10):
        distanceUpdate(Vectors, clusterCenters, distances)
        if (clusterUpdate(distances, cluster) < 100):
            break
        centerUpdate(Vectors, clusterCenters, cluster)
    print(i)
    return cluster

def AccuracyCalc(A,P):
    if (len(A)!=len(P)):
        return "Error: Predicted and Actual vectors are not of the same length!"
    Tr = 0
    for i in range(len(A)):
        if A[i]==P[i]:
            Tr += 1
    return Tr/len(A)*100

def mainFun():
    #k = int(input("Please enter number of expected clusters."))
    #Vectors = createVector()
    #clusterCenters = createVector()
    Vectors  = [[7.2,6.6,7.9,7.3,8,6.8,5,7,7.1],[3,3.1,3.2,2.6,3.9,3.3,4,2.8,1],[3.1,3.2,2.6,3.9,3.3,4,2.8,1,3],
        [3.2,2.6,3.9,3.3,4,2.8,1,3,3.1],[2.6,3.9,3.3,4,2.8,1,3,3.1,3.2],[7,7.1,7.2,6.6,7.9,7.3,8,6.8,5]
        ,[7.1,7.2,6.6,7.9,7.3,8,6.8,5,7],[10.6,11.9,11.3,12,10.8,9,11,11.1,11.2],[6.6,7.9,7.3,8,6.8,5,7,7.1,7.2]
        ,[11,11.1,11.2,10.6,11.9,11.3,12,10.8,9],[11.1,11.2,10.6,11.9,11.3,12,10.8,9,11],
        [11.2,10.6,11.9,11.3,12,10.8,9,11,11.1]]
    Actual = [2,1,1,1,1,2,2,3,2,3,3,3]
    Centro = [[3,3,3,3,3,3,3,3,3],[7,7,7,7,7,7,7,7,7],[11,11,11,11,11,11,11,11,11]]
    # It is better to assign cluster centers by user to aviod any mistakes, since it is a supervised process
    predicted = clustring (Vectors, Centro)
    print(clustring (Vectors, Centro))
    acc = AccuracyCalc(Actual, predicted)
    print("Accuracy is: ",acc,"%")  

def InitialCluster(Vectors,Lables,Factors):
    clusterCenters = createZeroVector(len(Factors),len(Vectors[0]))
    for i in range(len(Vectors[0])):
        sum = [0.0,0.0,0.0] #sum = createZeroVector(1,len(Factors))
        numNodes = [0,0,0]  #numNodes = createZeroVector(1,len(Factors))
        for j in range(len(Vectors)):
            for k in range(len(sum)):
                if Lables[j] == Factors[k]:
                   sum[k] += Vectors[j][i]
                   numNodes[k] += 1
        for l in range(len(sum)):
            if (numNodes[l] == 0):
                numNodes[l] = 1
            clusterCenters[l][i] = sum[l]/numNodes[l]
    return clusterCenters

def deleteOut (Vector):
    for row in Vector:
        del row[0]
    return Vector

def Actually (Vector):
    return [row[0] for row in Vector]

def toNum (twoDVector):
    return[[float(y) for y in x] for x in twoDVector]

def toFactor(list):
    vector = [0 for x in range(len(list))] 
    for ele in range(len(list)):
        if list[ele] == "EAP":
            vector[ele] = 1
        elif list[ele] == "HPL":
            vector[ele] = 2
        elif list[ele] == "MWS":
            vector[ele] = 3
        else:
            vector[ele] = 0
    return vector

import csv
data = list(csv.reader(open("datasetNorL.csv")))
del data[0]
split = len(data)*4//5
trainset = data[:split]
testset = data[split:]

lables = Actually(trainset)

Train = deleteOut(trainset)
Train = toNum(Train)

Centro = InitialCluster(Train,lables,['EAP','HPL','MWS'])
Actual = Actually(testset)
newww = toFactor(Actual)
Vectors = deleteOut(testset)
Vectors = toNum(Vectors)
predicted = clustring (Vectors, Centro)
acc = AccuracyCalc(newww, predicted)
print("Accuracy is: ",acc,"%")
#mainFun()
