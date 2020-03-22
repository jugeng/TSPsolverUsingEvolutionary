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

from datetime import datetime

#Controller Variables
data = "test_data"
populationSize = 100
mutationRate = 0.1
genCount = 500
numberOfCities = 0


#Calculators
totalFitness = 0

#Result Store
minDist = math.inf
bestRoute = []

#Data-plotting
fitness_curve = []


def addCity_using_coords():
    global numberOfCities, cityCoord
    
    temp_list = []
    with open(str(data_fname), "r") as f:
        for line in f:
            x, y = line.split()
            temp_list.append([float(x),float(y)])  #Convert to float for accuracy
        cityCoord = pd.DataFrame(temp_list, columns = ["x-coord", "y-coord"])       #Initiating pandas dataframe
    numberOfCities =  len(cityCoord) 
    if numberOfCities > 0:
        log.write("Successfully added {cit} cities from data.\n".format(cit = numberOfCities))
    #print(cityCoord, numberOfCities)

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
    pop = np.append(pop, [0], axis=0)
  
    populationMatrix.loc[len(populationMatrix)] = pop
  
    for i in range(populationSize-1):
        np.random.shuffle(pop[1:])
        populationMatrix.loc[len(populationMatrix)] = pop

    log.write("Initial Population Generated.\n")
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
    global totalFitness, fitnessMatrix, minDist, fitness_curve, bestRoute, nextGenerationMatrix

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
    fitnessMatrix = np.divide(fitnessMatrix,totalFitness)       #Normalizing the fitness values between [0-1]
    nextGenerationMatrix = nextGenerationMatrix.append(individual)    #Elitism, moving the fittest gene to the new generation as is
    print(minDist)


def calculateDistance(loc_1, loc_2):
    return distanceMatrix.iat[loc_1,loc_2]


def mutateChild(gene):
    global mutationRate
    r = random.random()
    if r < mutationRate:
        return mutation.RSM(gene)

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

        while (len(newGen)!= populationSize-2):
            parentA = matingPoolSelection()
            parentB = matingPoolSelection()
            childA, childB = crossover.cycleCrossover(parentA, parentB)
            mutateChild(childA)
            mutateChild(childB)
            newGen.append(childA)
            newGen.append(childB)

       
        nextGenerationMatrix = nextGenerationMatrix.append(newGen)
        populationMatrix.update(nextGenerationMatrix)
        nextGenerationMatrix = nextGenerationMatrix.iloc[0:0]
        calculateFitness()

        if(minDist == m):
            counter += 1 
        else:
            counter = 0 

        if (counter == 20):
            log.write("GENERATIONS EVOLVED={gen}\n".format(gen=i+1))
            break  
    #log.write("GENERATIONS EVOLVED={gen}\n".format(gen=i+1))   #Enable if a stopping condition is maintained  
        


#Data Logging
fname = "./logs/test_log/TSP_" + str(populationSize) + "_" + str(mutationRate) + "_"+ datetime.now().strftime("%d-%m-%y %H_%M") + ".txt"
log = open(str(fname), "w")
log.write("TSP USING GA\nDeveloped by Jugen Gawande\nRun Test ")
log.write(str(datetime.now()))
log.write("\nPOPULATION SIZE={pop} \nMUTATION RATE={mut} \n".format(pop =populationSize, mut = mutationRate))



data_fname = data + ".txt"   #Select data file to use
addCity_using_coords()


#Initialize pandas dataframes
distanceMatrix = pd.DataFrame(columns = np.arange(numberOfCities))
populationMatrix = pd.DataFrame(columns=np.arange(numberOfCities))
nextGenerationMatrix = pd.DataFrame(columns=np.arange(numberOfCities))

#Run Genetic Algorithm
generateDistMatrix()
generateInitPop()
nextGeneration()


log.write("Algorithm Completed.\n")
log.write("MINIMAL DISTANCE={}\n".format(minDist))
log.write("BEST ROUTE FOUND={}\n".format(bestRoute))
log.close()

with open("./logs/FC_{}_{}_{}.csv".format(populationSize,mutationRate,datetime.now().strftime("%d-%m-%y %H_%M")), "w") as f:
    f.write(str(fitness_curve))


#Graphing 
import matplotlib 
import matplotlib.pyplot as plt

plt.figure(1)
x = np.arange(len(fitness_curve))
y = fitness_curve
plt.title(data)
plt.plot(x,y)
plt.xlabel('Generations')
plt.ylabel('Distance')

plt.figure(2)
x_co=[]
y_co=[]
plt.scatter(cityCoord.iloc[:,0], cityCoord.iloc[:,1])
for i in bestRoute:
    x_co.append(cityCoord.iloc[i,0])
    y_co.append(cityCoord.iloc[i,1])
plt.plot(x_co,y_co)


plt.show()
plt.savefig("./logs/G_{}_{}_{}.png".format(populationSize,mutationRate,datetime.now().strftime("%d-%m-%y %H_%M")))


print(minDist, bestRoute)
print("-------------Done!------------")


