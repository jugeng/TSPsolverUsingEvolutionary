import random
import numpy as np

def OC_Single(parentA, parentB):
    geneCount = len(parentA) 

    r = random.randint(1, geneCount-1)
    childA = [] 
    childB = []

    childA.append(parentB[:r])
    childB.append(parentA[:r])
    print(type(childA))
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

    childA = [0]
    childB = [0]

    childA.append(parentB[1])
    a = 1
    while parentA[1] not in childB:
        a = np.where(parentA == parentB[a])
        a = np.where(parentA == parentB[a])
        
        
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




