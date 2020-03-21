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

    for gene in np.nditer (parentB, flags=["refs_ok"]):
        if gene in childB:
            continue
        else:
            childB.append(int(gene))
    return childA, childB

