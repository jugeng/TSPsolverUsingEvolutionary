# Developed by Jugen Gawande
# This is a python script to solve Travelling Salesman Problem using an evolutionary optimization
# algorithm called Genetic Algorithm.

import numpy as np
import random
import math
import pandas as pd

from os import system

from datetime import datetime
import logging
import sys
from time import time
import eel



#Calculators
numberOfCities = 0
alpha_temp = 0.9

loc_multiplier = math.pi / 180

#Result Store
minDist = math.inf
bestRoute = []

#Data-plotting
fitness_curve = []

data_cordinate = True

#Performance Measure
s_t = 0.0
e_t = 0.0

eel.init("renderer")


@eel.expose
def addCity_using_coords(arr):

    global numberOfCities, cityCoord,distanceMatrix, s_t, e_t

    s_t = time()

    temp_list = []
    for city in arr:
       temp_list.append([float(city[0]),float(city[1])])   #Convert to float for accuracy
    
    cityCoord = pd.DataFrame(temp_list, columns = ["x-coord", "y-coord"])       #Initiating pandas dataframe
    numberOfCities =  len(cityCoord)

    
    if numberOfCities > 0:
        distanceMatrix = pd.DataFrame(columns = np.arange(numberOfCities))
        logger.info("Successfully added {cit} cities from data.".format(cit = numberOfCities))
    
        generateDistMatrix()
    else:
        logger.warning("Problem loading cities")

    
    #print(cityCoord, numberOfCities)


def addCity_using_dist(arr):

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
        for j in range(numberOfCities):  #Generating entire matrix. Can be optimized by generating upward triangle matrix
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

        distanceMatrix.loc[len(distanceMatrix)] = temp_dist

    e_t = time()
    logger.info("CPU took {} to complete data loading and distance matrix building".format(e_t-s_t))
    #print(distanceMatrix)


#Data Logging
def logging_setup():
    global logger

    logger = logging.getLogger('tsp_sa')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    logger.info("TSP USING Simulated Annealing Algorithm\nDeveloped by Jugen Gawande")
    logger.info(str(datetime.now()))
    


def calculateSolutionFitness(arr):
    distance = 0
    for j in range(len(arr)-1):
        distance += distanceMatrix.iat[arr[j],arr[j+1]]

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
    b = random.randint(a+1, size-1)
    newarr = np.array([])
    if r > 0.5:
        return (reverse(arr, a, b))
    else:
        return (transport(arr, a, b))


def SA(arr):
    global T, minDist, bestRoute, s_t, e_t

    accepted = 1
    s_t = time()


    while(accepted != 0):
    
        accepted = 0

        for i in range(100 * len(arr)):
            
            new_arr = testNeighbor(arr)
        
            if(np.array_equal(new_arr, arr) == False):
                accepted += 1
                arr = new_arr

            if(accepted > 10 * len(arr)):
                break
            

        distance = calculateSolutionFitness(arr)
        logger.info("Temp: {} Dist:{} Accepted:{}".format(T,minDist,accepted))        

        T *= alpha_temp

        if(minDist > distance):
            minDist = distance
            bestRoute = arr
        
        # bestRoutestr = ' '.join([str(elem) for elem in arr]) 
        # eel.update_distance(round(distance, 2), bestRoutestr)()
        

    e_t = time()
    logger.info("CPU execution time: {}".format(e_t-s_t))

    return (calculateSolutionFitness(arr), arr)


    
@eel.expose 
def runAlgo(temp):

    global T, minDist, bestRoute

    T =  temp
    logger.info("\nTEMPURATURE={temp}".format(temp =T))

    route = np.arange(numberOfCities)
    bestRoute = route

    if(len(route) == numberOfCities):
        logger.info("Initial solution generated successfully")
    else:
        logger.warning("Problem generating initial population")
        sys.exit()

    minDist,bestRoute = SA(route)

    logger.info("MINIMAL DISTANCE={}".format(minDist))
    logger.info("BEST ROUTE FOUND={}".format(bestRoute))
    logger.info("\nAlgorithm Completed Successfully.")

    return minDist

logging_setup()
eel.start("index.html", size=(1600,850))