#Module to perform Mutation in Genetic Algorithm 
import random
from math import ceil

def Twors(individual):
    #Simple swap mutation where to genes are swapped to create a new gene
    size = len(individual)
    a = random.randint(1,size-3)
    b = random.randint(a+1, size-2)
    individual[a], individual[b] =  individual[b], individual[a]
    return (individual)


def RSM(individual):
    #Reverse Sequence Mutation: A subset of the individual is reversed to produce variation
    size = len(individual)
    a = random.randint(1,size-4)
    b = random.randint(a, size-2)
    # a = random.randint(1,int(size/4)-1)
    # b = random.randint(a, int(size/4))
    
    
    for i in range(ceil((b-a)/2)):
        individual[a+i], individual[b-i] = individual[b-i], individual[a+i]
    
    return (individual)
