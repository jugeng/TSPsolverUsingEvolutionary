import random
import numpy as np

def OC_Single(parentA, parentB):
    geneCount = len(parentA) 

    r = random.randint(2, geneCount-1)
    childA = list(parentB[:r])
    childB = list(parentA[:r])

    for gene in np.nditer(parentA, flags=["refs_ok"]):

        if gene in childA:
            continue
        else:
            childA.append(int(gene))

    for gene in np.nditer(parentB, flags=["refs_ok"]):

        if gene in childB:
            continue
        else:
            childB.append(int(gene))

  
    return childA, childB

def cycleCrossover(parentA, parentB):
    geneCount = len(parentA)

    childA = []
    childB = []

    childA.append(parentB[0])
    childB.append(parentA[0])
    childA.append(parentB[1])
    a = 1
    while parentA[1] not in childB:
        a = np.where(parentA == parentB[a])
        a = np.where(parentA == parentB[a])
        #print(parentB[a], "->", parentA[a])
        
        childB.append(int(parentB[a]))

        a = np.where(parentA == parentB[a])

        if parentB[a] not in childA:
            childA.append(int(parentB[a]))

    if len(childA) != geneCount:
        for gene in np.nditer(parentA, flags=["refs_ok"]):
            if gene in childA:
                continue
            else:
                childA.append(int(gene))
    
    if len(childB) != geneCount:
        for gene in np.nditer(parentB, flags=["refs_ok"]):
            if gene in childB:
                continue
            else:
                childB.append(int(gene))

    return childA, childB

def PMS(parentA, parentB):
    geneCount = len(parentA)

    childA = []
    childB = []

    a = random.randint(1,geneCount-2)
    b = random.randint(a+1, geneCount-1)

    print(a,b)
    print(parentA, parentB)

    childA[:a] = parentA[:a]
    childA[b-1:] = parentA[b-1:]
    childB[:a] = parentB[:a]
    childB[b-1:] = parentB[b-1:]

    childA[a:b+1] = parentB[a:b+1]
    childB[a:b+1] = parentA[a:b+1]
    
    mapping = []
    for i in range(a,b+1):
        mapping.append(parentB[i])
        mapping.append(parentA[i])

    print(mapping)
    
    return (childA, childB)





def Partial(parentA, parentB):
    firstCrossPoint = np.random.randint(0,len(parentA)-2)
    secondCrossPoint = np.random.randint(firstCrossPoint+1,len(parentA)-1)

    print(firstCrossPoint, secondCrossPoint)

    parentAMiddleCross = parentA[firstCrossPoint:secondCrossPoint]
    parentBMiddleCross = parentB[firstCrossPoint:secondCrossPoint]

    temp_childA = parentA[:firstCrossPoint] + parentBMiddleCross + parentA[secondCrossPoint:]

    temp_childB = parentB[:firstCrossPoint] + parentAMiddleCross + parentB[secondCrossPoint:]

    relations = []
    for i in range(len(parentAMiddleCross)):
        relations.append([parentBMiddleCross[i], parentAMiddleCross[i]])

    print(relations)

    childA=recursion1(temp_childA,firstCrossPoint,secondCrossPoint,parentAMiddleCross,parentBMiddleCross)
    childB=recursion2(temp_childB,firstCrossPoint,secondCrossPoint,parentAMiddleCross,parentBMiddleCross)

    return (childA, childB)


def recursion1 (temp_child , firstCrossPoint , secondCrossPoint , parentAMiddleCross , parentBMiddleCross) :
    child = np.array([0 for i in range(len(parentA))])
    for i,j in enumerate(temp_child[:firstCrossPoint]):
        c=0
        for x in relations:
            if j == x[0]:
                child[i]=x[1]
                c=1
                break
        if c==0:
            child[i]=j
    j=0
    for i in range(firstCrossPoint,secondCrossPoint):
        child[i]=parentBMiddleCross[j]
        j+=1

    for i,j in enumerate(temp_child[secondCrossPoint:]):
        c=0
        for x in relations:
            if j == x[0]:
                child[i+secondCrossPoint]=x[1]
                c=1
                break
        if c==0:
            child[i+secondCrossPoint]=j
    child_unique=np.unique(child)
    if len(child)>len(child_unique):
        child=recursion1(child,firstCrossPoint,secondCrossPoint,parentAMiddleCross,parentBMiddleCross)
    return(child)

def recursion2(temp_child,firstCrossPoint,secondCrossPoint,parentAMiddleCross,parentBMiddleCross):
    child = np.array([0 for i in range(len(parentA))])
    for i,j in enumerate(temp_child[:firstCrossPoint]):
        c=0
        for x in relations:
            if j == x[1]:
                child[i]=x[0]
                c=1
                break
        if c==0:
            child[i]=j
    j=0
    for i in range(firstCrossPoint,secondCrossPoint):
        child[i]=parentAMiddleCross[j]
        j+=1

    for i,j in enumerate(temp_child[secondCrossPoint:]):
        c=0
        for x in relations:
            if j == x[1]:
                child[i+secondCrossPoint]=x[0]
                c=1
                break
        if c==0:
            child[i+secondCrossPoint]=j
    child_unique=np.unique(child)
    if len(child)>len(child_unique):
        child=recursion2(child,firstCrossPoint,secondCrossPoint,parentAMiddleCross,parentBMiddleCross)
    return(child)


print (Partial(np.array([1,2,4,5,6,7,3]),np.array([3,5,2,6,4,7,1])))