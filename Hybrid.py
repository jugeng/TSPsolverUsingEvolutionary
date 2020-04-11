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
ex_time = 0.0
scale_factor = 0.000125



def addCity_using_coords():

    global numberOfCities, cityCoord,distanceMatrix, s_t, e_t

    s_t = time()
    cityCoord = []

    try:
        with open(str(data_fname), "r") as f:
            for line in f:
                i, x, y = line.split()
                cityCoord.append([float(x)*scale_factor,float(y)*scale_factor])  #Convert to float for accuracy

        numberOfCities =  len(cityCoord)

        if numberOfCities > 0:
            distanceMatrix = []
            logger.info("Successfully added {cit} cities from data.".format(cit = numberOfCities))
            generateDistMatrix()

    except:
        logger.warning("Dataset could not be loaded")
        sys.exit()


def addCity_using_dist():

    global distanceMatrix, numberOfCities, s_t, e_t

    s_t = time()
    dist_row = []

    with open(str(data_fname), "r") as f:

   
        for line in f:
            for val in line.split():
                dist_row.append(float(val))
    
            distanceMatrix.append (dist_row)
            dist_row = []

        numberOfCities = len(distanceMatrix[0])


    logger.info("Successfully added {cit} cities from data.".format(cit = numberOfCities))

    e_t = time()
    logger.info("CPU took {} to complete data loading and distance matrix building".format(e_t-s_t))


def generateDistMatrix():

    global distanceMatrix, s_t, e_t
    global numberOfCities


    def deg2rad(deg):
        return deg * loc_multiplier


    for i in range(numberOfCities):
        dist_row = []

        for j in range(i):  #Generating entire matrix. Can be optimized by generating upward triangle matrix
            a = cityCoord[i]
            b = cityCoord[j]

            #Find Euclidean distance between points

            if (data_cordinate == True):
                distance = math.sqrt(math.pow(b[0]-a[0], 2) + math.pow(b[1] - a[1], 2 ))
                dist_row.append(float(distance))   #Using python list comprehension for better performance

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

                dist_row.append(float(distance))

        distanceMatrix.append (dist_row)

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

    # pop = np.arange(numberOfCities)
    # bestRoute = pop 
    # populationMatrix.loc[len(populationMatrix)] = pop

    # for i in range(populationSize-1):
    #     worker = Thread(target=findChromo, args=(pop,))
    #     worker.start()
    #     threads.append(worker)

    # for t in threads:
    #     t.join()

    # for c in temp_pop:
    #     populationMatrix.loc[len(populationMatrix)] = c


    pop = list(range(numberOfCities))
    pop.append(0)

    for i in range(populationSize):
        pop.pop(0)
        pop.pop()

        pop_t = random.sample(pop, len(pop))

        pop_t.append(0)
        pop_t.insert(0,0)

        populationMatrix.append(pop_t)
        pop = pop_t.copy()


    logger.info("{} intial chromorsome populated".format(len(populationMatrix)))

    if (len(populationMatrix) == populationSize):
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

    return populationMatrix[index].copy()


def calculateFitness():
    global totalFitness, fitnessMatrix, minDist, fitness_curve, bestRoute, generation_fitness

    fitnessMatrix =[]

    for individual in populationMatrix:
   
        distance = 0
        for j in range(len(individual)-1):

            if(individual[j] < individual[j+1]):
                distance += distanceMatrix[individual[j+1]] [individual[j]]
            else:
                distance += distanceMatrix [individual[j]] [individual[j+1]]

        fitnessMatrix.append( 1 / distance )  #For routes with smaller distance to have highest fitness

        #Updating the best distance variable when a distance that is smaller than all
        #previous calculations is found

        if distance < minDist:
            minDist = distance
            bestRoute = individual.copy()


    totalFitness = sum(fitnessMatrix)
    fitnessMatrix = [x / totalFitness for x in fitnessMatrix]      #Normalizing the fitness values between [0-1]

    fitness_curve.append(round(minDist  / scale_factor, 2))
    #generation_fitness.loc[len(generation_fitness)] = fitnessMatrix


def mutateChild(gene):
    global mutationRate

    r = random.random()

    if r < mutationRate:

        return mutation.RSM(gene)


def nextGeneration():
    global nextGenerationMatrix, populationMatrix, genCount, bestRoute
  
    nextGenerationMatrix.append (bestRoute)     #Elitism, moving the fittest gene to the new generation as is
    nextGenerationMatrix.append (bestRoute)

    while (len(nextGenerationMatrix) < populationSize-2):

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

            nextGenerationMatrix.append(childA)
            nextGenerationMatrix.append(childB)
        else:
            nextGenerationMatrix.append(parentA)
            nextGenerationMatrix.append(parentB)

    populationMatrix = nextGenerationMatrix.copy()
    nextGenerationMatrix.clear()
    calculateFitness()


def calculateSolutionFitness(arr):
    distance = 0
    for j in range(len(arr)-1):

        if(arr[j] < arr[j+1]):
            distance += tempDistMatx[arr[j+1]][arr[j]]
        else:
            distance += tempDistMatx[arr[j]][arr[j+1]]

    return (distance)



def reverse(arr, a, b):

    x = arr[:a].copy()
    y = arr[a:b+1].copy()
    z = arr[b+1:].copy()

    w = y[::-1].copy()

    if (b != len(arr)-1):

        s = calculateSolutionFitness([arr[a-1], *y, arr[b+1]])
        s_dash = calculateSolutionFitness([arr[a-1], *w , arr[b+1]])

    else:
        s =  calculateSolutionFitness([arr[a-1], *y])
        s_dash = calculateSolutionFitness([arr[a-1],*w])

    del_e = s_dash - s

    if(del_e < 0):
  
        return ([*x, *w, *z])

    elif(del_e > 0):
        pr = math.exp((-del_e) / (T) )
        a = random.random()

        if (pr > a):
            return ([*x, *w, *z])

        else: return (arr)
    else: return(arr)

def transport(arr, a, b):

    x = arr[:a].copy()
    y = arr[a:b+1].copy()
    z = arr[b+1:].copy()

    m = arr[a-1:b+1].copy()

    s = calculateSolutionFitness(m)

    if((b-a) > len(arr)-2):
        if(b != len(arr)-1):
            u = random.randint(0,len(z)-1)
            if (u == 0):
                s_dash = calculateSolutionFitness([arr[a-1], *arr[a:b+1],z[u]]) 
            else:
                s_dash = calculateSolutionFitness([z[u-1], *arr[a:b+1], z[u]])
        else:
            u = random.randint(1,len(x)-1)
            s_dash = calculateSolutionFitness([arr[u], *arr[a:b+1], arr[u+1]])

    else:
        return(arr)

    del_e = s_dash - s

    if(del_e < 0):
        return [*x, *z[:u], *y, *z[u:]]

    elif(del_e > 0):
        pr = math.exp((-del_e) / (T) )
        a = random.random()

        if (pr > a):
            return [*x, *z[:u], *y, *z[u:]]

        else: return (arr)

    else: return(arr)


def testNeighbor(arr):
    r = random.random()
    size = len(arr)

    a = random.randint(1,size-4)
    b = random.randint(a+1, size-2)

    newarr = []

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
    while(k != 50):
        accepted = 0
        # for i in range(50*len(arr)):
        for i in range(100*count):
            new_arr = testNeighbor(arr)

            if(new_arr != arr):
                accepted += 1
                arr = new_arr.copy()

        T *= alpha_temp
        k += 1

    ret1.value = calculateSolutionFitness(arr)
    ret2[:] = arr
    

    # return(arr)


def GA():
    global nextGenerationMatrix, populationMatrix, bestRoute, dead_count, genEvolved,s_t, e_t, switch, minDist, ex_time
    global res, res_arr

    counter = 0
    i=0

    end = False

    n = minDist
    b = bestRoute

    t = 0.001
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
            print("\r",minDist / scale_factor, end = "\r")
            counter = 0


        if(counter == int(dead_count / 4) and switch == False):
         
            t = (1/i)
            sa = Process(target = SA, args=(b,n,t,1, distanceMatrix, res, res_arr))
            switch = True
            sa.start()

        #Reached End
        if (counter == dead_count and switch == False):
            #end = True
            t = (1/i)
            sa = Process(target = SA, args=(b,n,t,1, distanceMatrix, res, res_arr))
            switch = True
            sa.start()

        #Reached End
        if(counter >= dead_count and switch == True):
            sa.join()
            switch = False
            if(res.value < minDist):
                minDist = res.value
                bestRoute = res_arr[:]
                print("Inserted: ", minDist / scale_factor)
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

    fig = plt.figure(figsize = (23,8))
    fig.set_facecolor('#05070A')
    ax = fig.add_subplot(1, 2, 1)

    # decreasing time
    ax.set_xlabel('Generation', fontname="Calibri",fontweight="bold", fontsize=14)
    ax.set_ylabel('Distance', fontname="Calibri",fontweight="bold", fontsize=14)

    ax.spines['bottom'].set_color('#FFFAFF')
    ax.spines['left'].set_color('#FFFAFF')
    ax.spines['top'].set_color('#05070A')
    ax.spines['right'].set_color('#05070A')

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1)

    ax.tick_params(axis='x', colors='#FFFAFF')
    ax.tick_params(axis='y', colors='#FFFAFF')

    ax.yaxis.label.set_color('#1B9AAA')
    ax.xaxis.label.set_color('#1B9AAA')
    ax.title.set_color('#EEC643')
    
    ax.set_facecolor('#05070A')

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

    ax.plot(x,y, color = ("#72B01D"), linewidth = 2)


    gap = math.ceil(genEvolved / 25)
    plt.xticks(np.arange(0, genEvolved, gap ))


    sol = fig.add_subplot(1, 2, 2)

    m = []
    n = []
    r = []
    e = []

    for i in cityCoord:
        m.append (i[0])
        n.append (i[1])

    for j in bestRoute:
        r.append(cityCoord[j] [0])
        e.append (cityCoord[j] [1])
    
  
    sol.spines['bottom'].set_color('#05070A')
    sol.spines['left'].set_color('#05070A')
    sol.spines['top'].set_color('#05070A')
    sol.spines['right'].set_color('#05070A')

    plt.xticks(x, " ")
    plt.yticks(y, " ")

    sol.set_facecolor('#05070A')

    sol.plot(r, e, color = ("#CC3363"), linewidth = 2)
    sol.scatter(m[1:],n[1:], color = "#F79824", marker = "^" )
    sol.scatter(m[0],n[0], color = "#F79824", marker = "s", s = 75)
    

    fig.text(0.92, 0.84, '[INFO]', color = '#86BBD8')
    fig.text(0.92, 0.8, 'MIN DIST={:.2f}'.format(minDist / scale_factor), color = '#F5F1E3')
    fig.text(0.92, 0.78, 'GEN EVOLVED={}'.format(genEvolved), color = '#F5F1E3')
    fig.text(0.92, 0.76, 'DATASET={}'.format(data), color = '#F5F1E3')
    fig.text(0.92, 0.74, 'POP SIZE={}'.format(populationSize), color = '#F5F1E3')
    fig.text(0.92, 0.72, 'MUT RATE={}'.format(mutationRate), color = '#F5F1E3')
    fig.text(0.92, 0.70, 'CROSSOVER OPT={}'.format(CONFIG['OPERATOR']['CROSSOVER_OPERATOR']), color = '#F5F1E3')
    fig.text(0.92, 0.68, 'TEMPERATURE={}'.format(CONFIG['SIMULATED ANNEALING']['TEMPERATURE']), color = '#F5F1E3')
    fig.text(0.70, 0.02, "TSP solved using Modified Genetic Algorithm using SA [Visualizer] {}".format(datetime.now()), color = '#86BBD8')

    c=0.95
    x = 0.01
    for i in range(len(x1)):
        if(c < 0.05):
            c = 0.95
            x = 0.055
        fig.text(x,c,"[{}]{}".format(x1[i],y1[i]), fontsize=8, color = "#FAC9B8" )
        c-=0.02

    plt.subplots_adjust(left=0.15)


    logger.info("Fitness Curve generated")
    if(set_debug == True):
        hjhsda = "./logs/output_curve/G_MGASA_{}_{}_{}.png".format(datetime.now().strftime("%d-%m-%y %H_%M"), data, round(minDist / scale_factor))
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

    rname = "./logs/MGASA.csv"

    try:
        with open(rname, "r") as f:
            for index, l in enumerate(f):
                pass
    except:
        index = -1

    with open(rname, "a") as f:
        f.write("Test {}, {}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(index+2 ,datetime.now(), data,CONFIG['OPERATOR']['CROSSOVER_OPERATOR'],CONFIG['OPERATOR']['MUTATION_OPERATOR'], populationSize, mutationRate, round ((minDist/ scale_factor),2), len(fitness_curve), round(ex_time, 4) ))

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
    p = numberOfCities + 1 
    res_arr = Array('i', p , lock=False )

    #Initialize pandas dataframes
    generation_fitness = pd.DataFrame(columns = np.arange(populationSize))
    populationMatrix = []
    nextGenerationMatrix = []

    #Run Genetic Algorithm
    s_t = time()

    generateInitPop()
    GA()

    e_t = time()
    ex_time = e_t-s_t
    #logger.info("FITNESS CURVE:\n{}".format(fitness_curve))
     
    logger.info("CPU execution time: {}".format(ex_time))
    logger.info("MINIMAL DISTANCE={}".format(minDist / scale_factor))
    logger.info("BEST ROUTE FOUND={}".format(bestRoute))
    logger.info("\nAlgorithm Completed Successfully.")
      #Will fail if all generations are exhausted

    if(set_debug == True):
        outputRecord()

    logger.info("All done")
