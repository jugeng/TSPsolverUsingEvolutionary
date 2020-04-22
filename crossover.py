import random
import numpy as np

def OC_Single(parentA, parentB):
    geneCount = len(parentA) 

    r = random.randint(1, geneCount-2)
    childA = [] 
    childB = []

    childA = list(parentB[:r])
    childB = list(parentA[:r])

    for gene in parentA:
        if gene in childA:
            continue
        else:
            childA.append(int(gene))

    for gene in parentB:
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

    a = random.randint(1,geneCount-4)
    b = random.randint(a+1, geneCount-2)
      
    childA[a:b+1] = parentB[a:b+1]
    childB[a:b+1] = parentA[a:b+1]
    
    mapping_a = list(parentB[a:b+1])
    mapping_b = list(parentA[a:b+1])
    
    def inList(val, arrFrom, arrTo):

        res = arrTo[arrFrom.index(val)]

        if(res in arrFrom):
            return (inList(res, arrFrom, arrTo))
        else: 
            return res
    
    p = b + 1

    for i in range(len(parentA)-len(mapping_a)):
        if(p >= len(parentA)): p = 0

        if (p > b):
            if(parentA[p] in mapping_a):
                childA.append(inList(parentA[p], mapping_a, mapping_b))
            else:
                childA.append(parentA[p]) 

            if(parentB[p] in mapping_b):
                childB.append(inList(parentB[p], mapping_b, mapping_a))
            else:
                childB.append (parentB[p])
    
        if(p < a):
            if(parentA[p] in mapping_a):
                childA.insert(p, inList(parentA[p], mapping_a, mapping_b) )
            else:
                childA.insert(p, parentA[p]) 

            if(parentB[p] in mapping_b):
                childB.insert(p, inList(parentB[p], mapping_b, mapping_a) )
            else:
                childB.insert(p, parentB[p])

        
        p +=1

    return (childA, childB)


def OC_Multi(parentA, parentB):
    geneCount = len(parentA)

    childA = []
    childB = []

    a = random.randint(1,geneCount-4)
    b = random.randint(a+1, geneCount-2)

    childA = list(parentA[a:b+1])
    childB = list(parentB[a:b+1])


    p = b  #pointer to track parent index
    c = p+1    #pointer to track child index
    d = p+1   #pointer to track child index

    for i in range(len(parentA)):
        p +=1
        
        if (p >= len(parentA)-1):p = -1
        if (c > len(parentA)-1): c = 0
        if (d > len(parentA)-1): d = 0

        if (c > b):
            if(parentB[p] not in childA):
                childA.append(parentB[p])
                c +=1
        

        if (d > b):
            if(parentA[p] not in childB):
                childB.append(parentA[p])
                d +=1
                

        if (c < a):
            if(parentB[p] not in childA):
                childA.insert(c, parentB[p])
                c +=1
        
        if (d < a):
            if(parentA[p] not in childB):
                childB.insert(d, parentA[p])
                d +=1
    
            

    return(childA, childB)



