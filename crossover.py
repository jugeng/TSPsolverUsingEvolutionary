import random
import numpy as np

def orderedCrossover_SingleCut(parentA, parentB):
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
    a = 0
    while parentA[0] not in childB:
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
