# Developed by Jugen Gawande 
# This is a python script to solve Travelling Salesman Problem using an evolutionary optimization
# algorithm called Genetic Algorithm. 

import numpy as np
import random
import math
import pandas as pd

from os import system

import crossover
import mutation

#Controller Variables
numberOfCities = 0
populationSize = 500
mutationRate = 0.4
genCount = 500


#Calculators
totalFitness = 0

#Result Store
minDist = math.inf
bestRoute = []

#Data-plotting
fitness_curve = []


def generateDistMatrix():

    global distanceMatrix
    global numberOfCities
    
    for i in range(numberOfCities):
        temp_dist = []
        for j in range(numberOfCities):  #Generating entire matrix. Can be optimized by generating upward triangle matrix
            a = cityCoord.iloc[i].values 
            b = cityCoord.iloc[j].values   
            #Find Euclidean distance between points
            distance = np.linalg.norm(a-b)
            temp_dist.append(float(distance))   #Using python list comprehension for better performance

        distanceMatrix.loc[len(distanceMatrix)] = temp_dist
    print(distanceMatrix)
  

def generateInitPop():
    global numberOfCities, populationSize
    
    pop = np.arange(numberOfCities)
    populationMatrix.loc[len(populationMatrix)] = pop
  
    for i in range(populationSize-1):
        np.random.shuffle(pop[1:])
        populationMatrix.loc[len(populationMatrix)] = pop

    calculateFitness()


def matingPoolSelection():
    #Using Roulette wheel selection we will assign probabilities from 0-1 to 
    #each individual. The probability will determine how often we select a fit individual
    
    global totalFitness, fitnessMatrix

    index = 0
    r = random.random()

    while( r > 0):
        r = r - fitnessMatrix[index]
        index += 1 
    
    index -= 1

    return populationMatrix.iloc[index].values
    

def calculateFitness():
    global totalFitness, fitnessMatrix, minDist, fitness_curve, bestRoute

    fitness =[]
    for i,individual in populationMatrix.iterrows():
        distance = 0
        for j in range(len(individual)-1):
            #distance += calculateDistance(individual[j],individual[j+1])
            distance += distanceMatrix.iat[individual[j],individual[j+1]]
   
        fitness.append( 1 / distance )  #For routes with smaller distance to have highest fitness

        #Updating the best distance variable when a distance that is smaller than all
        #previous calculations is found
        
        if distance < minDist:
            minDist = distance
            bestRoute = np.copy(individual)
            
    fitness_curve.append(minDist)
    fitnessMatrix = np.asarray(fitness)
    totalFitness = np.sum(fitnessMatrix)
    fitnessMatrix = np.divide(fitnessMatrix,totalFitness)
    print(minDist)


def calculateDistance(loc_1, loc_2):
    return distanceMatrix.iat[loc_1,loc_2]


def mutateChild(gene):
    global mutationRate
    r = random.random()
    if r < mutationRate:
        return mutation.Twors(gene)

def nextGeneration():
    global nextGenerationMatrix, populationMatrix, genCount, bestRoute

    counter = 0
    i=0
    while(1):
    #for i in range(genCount):
        i+=1

        _ = system('cls')  #refresh and clear terminal window

        print("Generation: ", i+1)

        m = minDist
        newGen = []

        while (len(newGen)!= populationSize):
            parentA = matingPoolSelection()
            parentB = matingPoolSelection()
            child = crossover.orderedCrossover_SingleCut(parentA, parentB)
            mutateChild(child)
            newGen.append(child)

       
        nextGenerationMatrix = nextGenerationMatrix.append(newGen)
        populationMatrix.update(nextGenerationMatrix)
        nextGenerationMatrix = nextGenerationMatrix.iloc[0:0]
        calculateFitness()

        if(minDist == m):
            counter += 1 
        else:
            counter = 0 

        if (counter == 100):
            break  
        
        
  
#Enter city co-ordinates into to the program. Using an external coords file to import data

temp_list = []
#with open("test_data.txt", "r") as f:
with open("lau15_xy.txt", "r") as f:
    for line in f:
        x, y = line.split()
        temp_list.append([float(x),float(y)])  #Convert to float for accuracy

cityCoord = pd.DataFrame(temp_list, columns = ["x-coord", "y-coord"])  #Initiating pandas dataframe

numberOfCities =  len(cityCoord) 
if numberOfCities > 0:
    print("Successfully added",numberOfCities, "cities from data.")
#print(cityCoord, numberOfCities)

distanceMatrix = pd.DataFrame(columns = np.arange(numberOfCities))
populationMatrix = pd.DataFrame(columns=np.arange(numberOfCities))
nextGenerationMatrix = pd.DataFrame(columns=np.arange(numberOfCities))

generateDistMatrix()
generateInitPop()
nextGeneration()

print(minDist, bestRoute)