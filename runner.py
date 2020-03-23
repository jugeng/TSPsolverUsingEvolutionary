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
import matplotlib 
import matplotlib.pyplot as plt

#Controller Variables
data = "test_data"
data_type_flag = 0
populationSize = 80
mutationRate = 0.09
genCount = 500
numberOfCities = 0
dead_count = 50


#Calculators
totalFitness = 0

#Result Store
minDist = math.inf
bestRoute = []

#Data-plotting
fitness_curve = []



def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    # Print iterations progress
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r{} {} |{}| {}% {} CURR_MIN_DIST={:.2f}'.format(prefix,iteration, bar, percent, suffix, minDist), end = printEnd )
    # Print New Line on Complete
    if iteration == total: 
        print("\n")


def addCity_using_coords():
    global numberOfCities, cityCoord,distanceMatrix

    temp_list = []
    with open(str(data_fname), "r") as f:
        for line in f:
            x, y = line.split()
            temp_list.append([float(x),float(y)])  #Convert to float for accuracy
        cityCoord = pd.DataFrame(temp_list, columns = ["x-coord", "y-coord"])       #Initiating pandas dataframe
    numberOfCities =  len(cityCoord) 
    if numberOfCities > 0:
        print("Number of cities=", numberOfCities)
        distanceMatrix = pd.DataFrame(columns = np.arange(numberOfCities))
        log.write("Successfully added {cit} cities from data.\n".format(cit = numberOfCities))
    #print(cityCoord, numberOfCities)


def addCity_using_dist():
    
    global distanceMatrix, numberOfCities 

    temp_dist = []
    with open(str(data_fname), "r") as f:
        numberOfCities = int(f.readline())
        distanceMatrix = pd.DataFrame(columns = np.arange(numberOfCities))
    
        i = 0
        for line in f:
            for val in line.split():
                temp_dist.append(float(val))
                i += 1
                if i == numberOfCities:

                    i = 0
                    distanceMatrix.loc[len(distanceMatrix)] = temp_dist
                    temp_dist = [] 


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
    
    pop = [0]
    for i in range(numberOfCities):

        sort_dist = distanceMatrix.iloc[i]
        sort_dist = sort_dist.sort_values(ascending=True)
        

        for index, row in sort_dist.iteritems():
            if row == 0.0:
                continue
            
            if index in pop:
                continue
            else:
                pop.append(index)
                break 
    
    pop = np.asarray(pop)
    populationMatrix.loc[len(populationMatrix)] = pop
 
    for i in range(populationSize-1):
        np.random.shuffle(pop[1:])
        populationMatrix.loc[len(populationMatrix)] = pop

    #print(populationMatrix)
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
 

def calculateDistance(loc_1, loc_2):
    return distanceMatrix.iat[loc_1,loc_2]


def mutateChild(gene):
    global mutationRate
    r = random.random()
    if r < mutationRate:
        return mutation.Twors(gene)

def nextGeneration():
    global nextGenerationMatrix, populationMatrix, genCount, bestRoute, dead_count

    counter = 0
    i=0
    end_point = dead_count
    printProgressBar(0, end_point, prefix = 'Generation:', suffix = 'Complete', length = 50)

    while(1):
    #for i in range(genCount):

        #_ = system('cls')  #refresh and clear terminal window
        m = minDist
        newGen = []

        while (len(newGen)!= populationSize-2):
            parentA = matingPoolSelection()
            parentB = matingPoolSelection()    
            childA, childB = crossover.OC_Single(parentA, parentB)
            mutateChild(childA)
            mutateChild(childB)
            newGen.append(childA)
            newGen.append(childB)

       
        nextGenerationMatrix = nextGenerationMatrix.append(newGen[:populationSize])
        populationMatrix.update(nextGenerationMatrix)
        nextGenerationMatrix = nextGenerationMatrix.iloc[0:0]
        calculateFitness()
    
        if(minDist == m):
            counter += 1 
        else:
            counter = 0
            end_point = i + dead_count 

        if (counter == dead_count):
            log.write("GENERATIONS EVOLVED={gen}\n".format(gen=i+1))
            #plt.savefig("./logs/Attribute_{}_{}_{}.png".format(datetime.now().strftime("%d-%m-%y %H_%M"),populationSize,mutationRate,))
            break 
        i+=1 
        printProgressBar(i, end_point , prefix = 'Generation:', suffix = 'Evolved', length = 40)
    #log.write("GENERATIONS EVOLVED={gen}\n".format(gen=i+1))   #Enable if a stopping condition is maintained  

#Graphing
def graphing(): 
    

    plt.figure(1)
    x = np.arange(len(fitness_curve))
    y = fitness_curve
    plt.title(data)
    plt.plot(x,y)
    plt.xlabel('Generations')
    plt.ylabel('Distance')
    plt.savefig("./logs/output_curve/G_{}_{}_{}.png".format(populationSize,mutationRate,datetime.now().strftime("%d-%m-%y %H_%M")))
    print("Fitness curve saved to file.")

    plt.figure(2)
    x_co=[]
    y_co=[]
    plt.scatter(cityCoord.iloc[:,0], cityCoord.iloc[:,1], c="r")

    for i in bestRoute:
        x_co.append(cityCoord.iloc[i,0])
        y_co.append(cityCoord.iloc[i,1])
    x_co.append(cityCoord.iloc[0,0])
    y_co.append(cityCoord.iloc[0,1])
    plt.plot(x_co,y_co)


    plt.show()    


#Data Logging
fname = "./logs/test_log/TSP_" + datetime.now().strftime("%d-%m-%y %H_%M") + "_" + str(populationSize) + "_" + str(mutationRate) + ".txt"
log = open(str(fname), "w")
log.write("TSP USING GA\nDeveloped by Jugen Gawande\nRun Test ")
log.write(str(datetime.now()))
log.write("\nPOPULATION SIZE={pop} \nMUTATION RATE={mut} \nDATASET SELECTED={name}\n".format(pop =populationSize, mut = mutationRate, name = data))

print("\nPOPULATION SIZE={pop} \nMUTATION RATE={mut} \nDATASET SELECTED={name}\n".format(pop =populationSize, mut = mutationRate, name = data))


data_fname = "./dataset/" + data + ".txt"   #Select data file to use

if data_type_flag == 0:
    addCity_using_coords()
else:
    addCity_using_dist()

#Initialize pandas dataframes

populationMatrix = pd.DataFrame(columns=np.arange(numberOfCities))
nextGenerationMatrix = pd.DataFrame(columns=np.arange(numberOfCities))


#Run Genetic Algorithm
if distanceMatrix.empty == True:
    generateDistMatrix()
generateInitPop()
nextGeneration()

log.write("Algorithm Completed.\n")
log.write("MINIMAL DISTANCE={}\n".format(minDist))
log.write("BEST ROUTE FOUND={}\n".format(bestRoute))
log.close()
print("Log file generated.")
print("\nMINIMAL DISTANCE={}\nROUTE MINIMIZED={}".format(minDist, bestRoute))

with open("./logs/curve_log/FitnessCurve_{}_{}_{}.csv".format(datetime.now().strftime("%d-%m-%y %H_%M"),populationSize,mutationRate), "w") as f:
    f.write(str(fitness_curve))

graphing()



print("Done!")
