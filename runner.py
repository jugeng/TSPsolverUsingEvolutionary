# Developed by Jugen Gawande 
# This is a python script to solve Travelling Salesman Problem using an evolutionary optimization
# algorithm called Genetic Algorithm. 

import numpy as np
import random
import math
import pandas as pd

numberOfCities = 0
populationSize = 50
mutationRate = 0.05


fitnessMatrix = []
totalFitness = 0

minDist = math.inf
bestRoute = []

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
    #print(distanceMatrix)
  

def generateInitPop():
    global numberOfCities, populationSize
    
    pop = np.arange(numberOfCities)
    populationMatrix.loc[len(populationMatrix)] = pop
  
    for i in range(populationSize-1):
        np.random.shuffle(pop[1:])
        populationMatrix.loc[len(populationMatrix)] = pop
    

    calculateFitness()


def selection():
    return True    


def calculateFitness():
    global totalFitness, fitnessMatrix

    for i in range(populationSize):
        individual = populationMatrix.iloc[i].values
        distance = 0
        for j in range(len(individual)-1):
            distance += calculateDistance(individual[j],individual[j+1])
             
        fitnessMatrix.append(1 / distance)              #For routes with smaller distance to have highest fitness
    fitnessMatrix = np.asarray(fitnessMatrix)
    totalFitness = np.sum(fitnessMatrix)
    fitnessMatrix = np.divide(fitnessMatrix,totalFitness)
    print(fitnessMatrix)
  

def calculateDistance(loc_1, loc_2):
    return distanceMatrix.iat[loc_1,loc_2]


def fitnessNormalize():
    return True

def crossover():
    return True

def mutation():
    return True


#Enter city co-ordinates into to the program. Using an external coords file to import data

temp_list = []
with open("test_data.txt", "r") as f:
    for line in f:
        x, y = line.split()
        temp_list.append([int(x),int(y)])  #Convert to float for accuracy

cityCoord = pd.DataFrame(temp_list, columns = ["x-coord", "y-coord"])  #Initiating pandas dataframe

numberOfCities =  len(cityCoord) 

#print(cityCoord, numberOfCities)

distanceMatrix = pd.DataFrame(columns = np.arange(numberOfCities))
populationMatrix = pd.DataFrame(columns=np.arange(numberOfCities))


generateDistMatrix()
generateInitPop()