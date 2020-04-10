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
import logging
import configparser
import matplotlib
import matplotlib.pyplot as plt
import sys
from time import time

from multiprocessing import Process, Value, Array
from  threading import Thread

from queue import Queue

T = 0.5
alpha_temp = 0.9


CONFIG = configparser.ConfigParser()
CONFIG.read('controller.ini')

#Calculators
numberOfCities = 0
totalFitness = 0
genEvolved = 0

loc_multiplier = math.pi / 180


#Result Store
minDist = math.inf
bestRoute = []

tempDistMatx = []
switch = False

#Data-plotting
fitness_curve = []



#Performance Measure
s_t = 0.0
e_t = 0.0
scale_factor = 0.000125



def addCity_using_coords():

    global numberOfCities, cityCoord,distanceMatrix, s_t, e_t

    s_t = time()


    temp_list = []
    try:
        with open(str(data_fname), "r") as f:
            for line in f:
                
                i, x, y = line.split()
                temp_list.append([float(x)*scale_factor,float(y)*scale_factor])  #Convert to float for accuracy
            cityCoord = pd.DataFrame(temp_list, columns = ["x-coord", "y-coord"])       #Initiating pandas dataframe
        numberOfCities =  len(cityCoord)
        if numberOfCities > 0:
            distanceMatrix = pd.DataFrame(columns = np.arange(numberOfCities))
            logger.info("Successfully added {cit} cities from data.".format(cit = numberOfCities))
            generateDistMatrix()
    except:
        logger.warning("Dataset could not be loaded")
        sys.exit()


    # print(cityCoord, numberOfCities)


def addCity_using_dist():

    global distanceMatrix, numberOfCities, s_t, e_t
    s_t = time()
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

    logger.info("Successfully added {cit} cities from data.".format(cit = numberOfCities))
    e_t = time()
    logger.info("CPU took {} to complete data loading and distance matrix building".format(e_t-s_t))


def generateDistMatrix():

    global distanceMatrix, s_t, e_t
    global numberOfCities


    def deg2rad(deg):
        return deg * loc_multiplier


    for i in range(numberOfCities):
        temp_dist = []
        for j in range(i):  #Generating entire matrix. Can be optimized by generating upward triangle matrix
            a = cityCoord.iloc[i].values
            b = cityCoord.iloc[j].values
            #Find Euclidean distance between points

            #distance = np.linalg.norm(a-b)
            if (data_cordinate == True):
                distance = np.linalg.norm(a-b)
                temp_dist.append(float(distance))   #Using python list comprehension for better performance
            else:
                R = 6371 #Radius of the earth in km
                lat1 = a[0]
                lat2 = b[0]
                long1 = a[1]
                long2 = b[1]
                dLat = deg2rad(lat2-lat1)
                dLon = deg2rad(long2-long1)
                a = math.sin(dLat/2) * math.sin(dLat/2) + math.cos(deg2rad(lat1)) * math.cos(deg2rad(lat2)) * math.sin(dLon/2) * math.sin(dLon/2)
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                distance = R * c
                temp_dist.append(float(distance))

        temp_dist.extend([0.0] * (numberOfCities - len(temp_dist))) #Padding
        distanceMatrix.loc[len(distanceMatrix)] = temp_dist

    e_t = time()
    logger.info("CPU took {} to complete data loading and distance matrix building".format(e_t-s_t))
    # print(distanceMatrix)




def findChromo(pop ):
    global temp_pop

    r = random.random()
    if(r > 0.5):
        pop = SA(pop,0.0, 0.0025, numberOfCities, distanceMatrix, res, res_arr)
    else:
        np.random.shuffle(pop[1:])
    temp_pop.append(pop)
    
    


threads = []

def generateInitPop():
    global numberOfCities, populationSize, threads

    pop = np.arange(numberOfCities)
    bestRoute = pop 
    populationMatrix.loc[len(populationMatrix)] = pop


    for i in range(populationSize-1):
        worker = Thread(target=findChromo, args=(pop,))
        worker.start()
        threads.append(worker)

    for t in threads:
        t.join()

    for c in temp_pop:
        populationMatrix.loc[len(populationMatrix)] = c

    logger.info("{} intial chromorsome populated".format(len(populationMatrix.index)))

    if (len(populationMatrix.index) == populationSize):
        logger.info("Initial population generated successfully")
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
    global totalFitness, fitnessMatrix, minDist, fitness_curve, bestRoute, nextGenerationMatrix, generation_fitness

    fitness =[]
    for i,individual in populationMatrix.iterrows():
        distance = 0
        for j in range(len(individual)-1):
            if(individual[j] < individual[j+1]):
                distance += distanceMatrix.iat[individual[j+1],individual[j]]
            else:
                distance += distanceMatrix.iat[individual[j],individual[j+1]]

        fitness.append( 1 / distance )  #For routes with smaller distance to have highest fitness

        #Updating the best distance variable when a distance that is smaller than all
        #previous calculations is found

        if distance < minDist:
            minDist = distance
            bestRoute = np.copy(individual)

    fitnessMatrix = np.asarray(fitness)
    totalFitness = np.sum(fitnessMatrix)
    fitnessMatrix = np.divide(fitnessMatrix,totalFitness)       #Normalizing the fitness values between [0-1]

    fitness_curve.append(round(minDist  / scale_factor, 2))
    generation_fitness.loc[len(generation_fitness)] = fitnessMatrix


def mutateChild(gene):
    global mutationRate
    r = random.random()
    if r < mutationRate:
        return mutation.RSM(gene)


def nextGeneration():
    global nextGenerationMatrix, populationMatrix, genCount, bestRoute
    newGen = []

    nextGenerationMatrix.loc[len(nextGenerationMatrix)] = bestRoute     #Elitism, moving the fittest gene to the new generation as is
    nextGenerationMatrix.loc[len(nextGenerationMatrix)] = bestRoute

    while (len(newGen) < populationSize-2):

        parentA = matingPoolSelection()
        parentB = matingPoolSelection()

        #Crossover Probability
        r = random.random()

        if r < 0.8:
            if (cx_opt == "OC_Single"):
                childA, childB = crossover.OC_Single(parentA, parentB)
            elif (cx_opt == "cycleCrossover"):
                childA, childB = crossover.cycleCrossover(parentA, parentB)
            elif (cx_opt == "OC_Multi"):
                childA, childB = crossover.OC_Multi(parentA, parentB)
            elif (cx_opt == "PMS"):
                childA, childB = crossover.PMS(parentA, parentB)
            else:
                logger.warning("Unknown crossover operator configured.")
                logger.warning("Model cannot be executed")
                sys.exit()

            mutateChild(childA)
            mutateChild(childB)
            newGen.append(childA)
            newGen.append(childB)
        else:
            newGen.append(parentA)
            newGen.append(parentB)


    nextGenerationMatrix = nextGenerationMatrix.append(newGen[:populationSize-1])
    populationMatrix = nextGenerationMatrix.copy()
    nextGenerationMatrix = nextGenerationMatrix.iloc[0:0]
    calculateFitness()


def calculateSolutionFitness(arr):
    distance = 0
    for j in range(len(arr)-1):
        if(arr[j] < arr[j+1]):
            distance += tempDistMatx.iat[arr[j+1],arr[j]]
        else:
            distance += tempDistMatx.iat[arr[j],arr[j+1]]

    return (distance)



def reverse(arr, a, b):
    x = arr[:a]
    y = arr[a:b+1]
    z = arr[b+1:]

    w = y[::-1]

    if (b != len(arr)-1):
        s = calculateSolutionFitness(np.concatenate(([arr[a-1]],y,[arr[b+1]])))
        s_dash = calculateSolutionFitness(np.concatenate(([arr[a-1]],w,[arr[b+1]])))
    else:
        s = calculateSolutionFitness(np.concatenate(([arr[a-1]],y)))
        s_dash = calculateSolutionFitness(np.concatenate(([arr[a-1]],w)))

    del_e = s_dash - s

    if(del_e < 0):
        return (np.concatenate((x,w,z)))

    elif(del_e > 0):
        pr = math.exp((-del_e) / (T) )
        a = random.random()

        if (pr > a):

            return (np.concatenate((x,w,z)))
        else: return (arr)
    else: return(arr)


def transport(arr, a, b):

    x = arr[:a]
    y = arr[a:b+1]
    z = arr[b+1:]

    m = arr[a-1:b+1]
    s = calculateSolutionFitness(m)

    if((b-a) > len(arr)-2):
        if(b != len(arr)-1):
            u = random.randint(0,len(z)-1)
            if (u == 0):
                s_dash = calculateSolutionFitness(np.concatenate(([arr[a-1]], arr[a:b+1], [z[u]])))
            else:
                s_dash = calculateSolutionFitness(np.concatenate(([z[u-1]], arr[a:b+1], [z[u]])))
        else:
            u = random.randint(1,len(x)-1)
            s_dash = calculateSolutionFitness(np.concatenate(([arr[u]], arr[a:b+1], [arr[u+1]])))

    else:
        return(arr)

    del_e = s_dash - s

    if(del_e < 0):
        return (np.concatenate((x,z[:u],y,z[u:])))

    elif(del_e > 0):
        pr = math.exp((-del_e) / (T) )
        a = random.random()

        if (pr > a):
            return (np.concatenate((x,z[:u],y,z[u:])))

        else: return (arr)

    else: return(arr)


def testNeighbor(arr):
    r = random.random()
    size = len(arr)
    a = random.randint(1,size-3)
    b = random.randint(a+1, size-2)
    newarr = np.array([])

    if r > 0.5:
        return (reverse(arr, a, b))

    else:
        return (transport(arr, a, b))


def SA(arr, val, t, endp, dist, ret1, ret2):
    global T, tempDistMatx

    T = t
    tempDistMatx = dist
    accepted = math.inf
    count = len(arr)
    accepted = count + 1

    # while(accepted >= endp):
    k=0
    while(k != 10):
        accepted = 0
        # for i in range(50*len(arr)):
        for i in range(50*count):
            new_arr = testNeighbor(arr)
            if(np.array_equal(new_arr, arr) == False):
                accepted += 1
                arr = new_arr

        T *= alpha_temp
        k += 1

    ret1.value = calculateSolutionFitness(arr)
    ret2[:] = arr
    
    return(arr)


def GA():
    global nextGenerationMatrix, populationMatrix, bestRoute, dead_count, genEvolved,s_t, e_t, switch, minDist
    global res, res_arr

    counter = 0
    i=0


    end = False

    n = minDist
    b = bestRoute
    

    print(minDist / scale_factor)
    res = Value('f', minDist, lock=False )
    res_arr = Array('i', numberOfCities, lock=False )
    t = 0.005
    sa = Process(target = SA, args=(b,n,t,numberOfCities/4, distanceMatrix, res, res_arr))
    switch = True
    sa.start()

    
    if(res.value < minDist):
        minDist = res.value
        bestRoute = res_arr[:]

    while(1):
        m = minDist

        n = minDist
        b = bestRoute

        nextGeneration()

        if(minDist == m):
            counter += 1

        else:
            print(minDist / scale_factor)
            counter = 0


        if(counter == int(dead_count / 4) and switch == False):
         
            t = (1/i)*10
            sa = Process(target = SA, args=(b,n,t,1, distanceMatrix, res, res_arr))
            switch = True
            sa.start()

        #Reached End
        if (counter == dead_count and switch == False):
            end = True
        #     t = (1/i)*10
        #     sa = Process(target = SA, args=(b,n,t,1, distanceMatrix, res, res_arr))
        #     switch = True
        #     sa.start()

        #Reached End
        if(counter >= dead_count and switch == True):
            sa.join()
            switch = False
            if(res.value < minDist):
                minDist = res.value
                bestRoute = res_arr[:]

                print("Inserted: ",minDist / scale_factor)

                counter = 0
            else: end = True


        if(switch == True):
            if(sa.is_alive() == False):
                switch = False
                if(res.value < minDist):
                    minDist = res.value
                    bestRoute = res_arr[:]

                    print("Inserted: ",minDist / scale_factor)

                    counter = 0

        if(end == True):
            genEvolved = len(fitness_curve)
            logger.info("\nGENERATIONS EVOLVED={gen}".format(gen=str(genEvolved)))
            
            break
        
        else:
            i+=1


#Graphing
def graphing():
    global fitness_curve, genEvolved

    fig = plt.figure(figsize = (15,8))
    ax = fig.add_subplot(1, 1, 1)

    # decreasing time
    ax.set_xlabel('Generation', fontname="Calibri",fontweight="bold", fontsize=14)
    ax.set_ylabel('Distance', fontname="Calibri",fontweight="bold", fontsize=14)

    ax.spines['bottom'].set_color('#FFFAFF')
    ax.spines['left'].set_color('#FFFAFF')
    ax.spines['top'].set_color('#1B2533')
    ax.spines['right'].set_color('#1B2533')

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)

    ax.tick_params(axis='x', colors='#FFFAFF')
    ax.tick_params(axis='y', colors='#FFFAFF')

    ax.yaxis.label.set_color('#1B9AAA')
    ax.xaxis.label.set_color('#1B9AAA')
    ax.title.set_color('#EEC643')
    fig.set_facecolor('#1B2533')
    ax.set_facecolor('#1B2533')

    ax.grid(True,linewidth = 0.1)

    plt.title("Fitness Evolution Curve", loc='center' ,fontname="Calibri",fontweight="bold", fontsize=18)


    x = np.arange(len(fitness_curve))
    y = fitness_curve

    x1 = [0]
    y1 = [fitness_curve[0]]
    for i in range(genEvolved-1):
        if fitness_curve[i] != fitness_curve[i+1]:
            y1.append(fitness_curve[i+1])
            x1.append(i+1)

    ax.scatter(x1,y1, color = "#F79824" )
    ax.plot(x,y, color = ("#CC3363"), linewidth = 2)
    #ax.barh(y1,x1, color = ("#DEF4C6"), height = 0.08, alpha = 0.2 )

    gap = math.ceil(genEvolved / 25)
    plt.xticks(np.arange(0, genEvolved, gap ))

    fig.text(0.8, 0.84, '[INFO]', color = '#86BBD8')
    fig.text(0.8, 0.8, 'MIN DIST={:.2f}'.format(minDist / scale_factor), color = '#F5F1E3')
    fig.text(0.8, 0.78, 'GEN EVOLVED={}'.format(genEvolved), color = '#F5F1E3')
    fig.text(0.8, 0.76, 'DATASET={}'.format(data), color = '#F5F1E3')
    fig.text(0.8, 0.74, 'POP SIZE={}'.format(populationSize), color = '#F5F1E3')
    fig.text(0.8, 0.72, 'MUT RATE={}'.format(mutationRate), color = '#F5F1E3')
    fig.text(0.8, 0.70, 'CROSSOVER OPT={}'.format(CONFIG['OPERATOR']['CROSSOVER_OPERATOR']), color = '#F5F1E3')
    fig.text(0.8, 0.68, 'TEMPERATURE={}'.format(CONFIG['SIMULATED ANNEALING']['TEMPERATURE']), color = '#F5F1E3')

    fig.text(0.54, 0.02, "TSP solved using Modified Genetic Algorithm using SA [Visualizer] {}".format(datetime.now()), color = '#86BBD8')

    logger.info("Fitness Curve generated")

    c=0.95
    x = 0.01
    for i in range(len(x1)):
        if(c < 0.1):
            c = 0.95
            x = 0.92
        fig.text(x,c,"[{}]{}".format(x1[i],y1[i]), fontsize=8, color = "#FAC9B8" )
        c-=0.02

    plt.subplots_adjust(left=0.15)


    if(set_debug == True):
        hjhsda = "./logs/output_curve/G_MGASA_{}_{}_{}.svg".format(datetime.now().strftime("%d-%m-%y %H_%M"), data, round(minDist / scale_factor))
        fig.savefig(hjhsda, facecolor=fig.get_facecolor(), edgecolor='none')
    logger.info("Fitness Curve exported to logs\nFile name: {}".format(hjhsda) )
    #plt.show()   #To view graph after generating

#Data Logging
def loggingSetup():
    global logger
    logger = logging.getLogger('tsp_hybrid')
    fname = "./logs/test_log/TSP_GA_" + datetime.now().strftime("%d-%m-%y %H_%M") + "_" + data +"_"+ str(populationSize) + "_" + str(mutationRate) + ".log"

    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    if(set_debug == True):
        fh = logging.FileHandler(fname)
        fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    if(set_debug == True):
        logger.addHandler(fh)
    logger.addHandler(ch)


    logger.info("TSP USING Genetic Algorithm\nDeveloped by Jugen Gawande")
    logger.info(str(datetime.now()))
    logger.info("\nPOPULATION SIZE={pop} \nMUTATION RATE={mut} \nDATASET SELECTED={name}".format(pop =populationSize, mut = mutationRate, name = data))


def outputRecord():
    global generation_fitness
    with open("./logs/visualize_data.txt", "w") as f:
        f.write(" ".join(str(item) for item in fitness_curve))
        f.write("\n")

    rname = "./logs/test_MGASA_{}.csv".format(data)

    try:
        with open(rname, "r") as f:
            for index, l in enumerate(f):
                pass
    except:
        index = -1

    with open(rname, "a") as f:
        f.write("Test {}, {}, {}, {}, {}, {}, {:.2f}\n".format(index+2 ,datetime.now(),CONFIG['OPERATOR']['CROSSOVER_OPERATOR'],CONFIG['OPERATOR']['MUTATION_OPERATOR'], populationSize, mutationRate, minDist ))

    logger.info("Test results recorded.")
    logger.info("Visualization Data Generated")

    generation_fitness = generation_fitness.round(5)
    generation_fitness.to_csv("./logs/curve_log/GenFit_MGASA_data_{}.csv".format(datetime.now().strftime("%d-%m-%y %H_%M")), header=None, index=None, sep=',', mode='a')
    generation_fitness.to_csv('./logs/visualize_data.txt', header=None, index=None, sep=' ', mode='a')

    graphing()

#Controller Variables
def initializeAlgorithm():
    global data, data_type_flag, populationSize, mutationRate, genCount, dead_count, cx_opt, set_debug, data_cordinate, data_fname

    data = CONFIG['DATASET']['FILE_NAME']
    data_type_flag = CONFIG.getint('DATASET', 'DATASET_TYPE')
    populationSize = CONFIG.getint('GENETIC', 'POP_SIZE')
    if (populationSize < 1):
        logger.warning("Population size not enough")
        sys.exit()
    mutationRate = CONFIG.getfloat('GENETIC', 'MUTATION_RATE')
    dead_count = CONFIG.getint('GENETIC', 'DEAD_COUNTER')
    cx_opt = CONFIG['OPERATOR']['CROSSOVER_OPERATOR']
    set_debug = CONFIG.getboolean('DEBUG', 'LOG_FILE')
    data_cordinate = CONFIG.getboolean('DATASET','CONTAINS_COORDINATES')

    data_fname = "./dataset/" + data + ".txt"



if __name__ == '__main__':

    initializeAlgorithm()
    loggingSetup()

    if data_type_flag == 0:
        addCity_using_coords()

    else:
        addCity_using_dist()


    res = Value('f', minDist, lock=False )
    res_arr = Array('i', numberOfCities, lock=False )

    #Initialize pandas dataframes
    generation_fitness = pd.DataFrame(columns = np.arange(populationSize))
    populationMatrix = pd.DataFrame(columns=np.arange(numberOfCities))
    nextGenerationMatrix = pd.DataFrame(columns=np.arange(numberOfCities))

    #Run Genetic Algorithm
    s_t = time()

    generateInitPop()
    GA()

    e_t = time()
    logger.info("CPU execution time: {}".format(e_t-s_t))

    logger.info("MINIMAL DISTANCE={}".format(minDist / scale_factor))
    logger.info("BEST ROUTE FOUND={}".format(bestRoute))
    logger.info("\nAlgorithm Completed Successfully.")
    logger.info("FITNESS CURVE:\n{}".format(fitness_curve[:genEvolved]))   #Will fail if all generations are exhausted


    if(set_debug == True):
        outputRecord()

    logger.info("All done")
