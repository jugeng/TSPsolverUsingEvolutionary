import random
import numpy as np

def orderedCrossover_SingleCut(parentA, parentB):
    geneCount = len(parentA)

    r = random.randint(2, geneCount-1)
    child = list(parentB[:r])
    

    for gene in np.nditer(parentA, flags=["refs_ok"]):

        if gene in child:
            continue
        else:
            child.append(int(gene))

  
    return child

