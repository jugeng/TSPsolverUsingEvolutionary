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
import configparser
import matplotlib
import matplotlib.pyplot as plt
import sys
from time import time
import eel

CONFIG = configparser.ConfigParser()
CONFIG.read('controller.ini')

#Calculators
numberOfCities = 0
alpha_temp = 0.9

loc_multiplier = math.pi / 180

#Result Store


#Data-plotting
fitness_curve = []

#Performance Measure
s_t = 0.0
e_t = 0.0
ex_time = 0.0
scale_factor = 0.000125


cityCoord = []

eel.init("renderer")

@eel.expose
def addCity_using_coords(arr):

    global numberOfCities, cityCoord,distanceMatrix, s_t, e_t

    s_t = time()
    cityCoord.clear()
 

    for city in arr:
            i, x, y = city
            cityCoord.append([float(x)*scale_factor,float(y)*scale_factor])  #Convert to float for accuracy
           

    numberOfCities =  len(cityCoord)
  
 

    if numberOfCities > 0:
        distanceMatrix = []
        logger.info("Successfully added {cit} cities from data.".format(cit = numberOfCities))
        generateDistMatrix()

    if(numberOfCities != len(arr)):
        logger.warning("Dataset could not be loaded")
        sys.exit()
    
    
 
    #print(cityCoord, numberOfCities)


def addCity_using_dist():

    global distanceMatrix, numberOfCities, s_t, e_t

    s_t = time()

    
    with open(str(data_fname), "r") as f:
        distanceMatrix = []
        dist_row = []

        for line in f:
            for val in line.split():
                dist_row.append(float(val)*scale_factor)

                
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
        temp_dist = []
        for j in range(i):  #Generating entire matrix. Can be optimized by generating upward triangle matrix
            a = cityCoord[i]
            b = cityCoord[j]
            #Find Euclidean distance between points

            #distance = np.linalg.norm(a-b)
            if (data_cordinate == True):
                distance = math.sqrt(math.pow(b[0]-a[0], 2) + math.pow(b[1] - a[1], 2 ))
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

        
        distanceMatrix.append (temp_dist)
    
    e_t = time()
    logger.info("CPU took {} to complete data loading and distance matrix building".format(e_t-s_t))
    #print(distanceMatrix)


#Data Logging
def logging_setup():
    global logger

    logger = logging.getLogger('tsp_sa')

    logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    logger.info("TSP USING Simulated Annealing Algorithm\nDeveloped by Jugen Gawande")
    logger.info(str(datetime.now()))



def calculateSolutionFitness(arr):
    distance = 0
    
    for j in range(len(arr)-1):
        if(arr[j] < arr[j+1]):
            distance += distanceMatrix[arr[j+1]][arr[j]]

        else:
            distance += distanceMatrix[arr[j]][arr[j+1]]

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


def SA(arr):
    global T, minDist, bestRoute, s_t, e_t, ex_time

    accepted = 1
    s_t = time()

    while(accepted != 0):
        
        accepted = 0

        for i in range(100 * len(arr)):

            new_arr = testNeighbor(arr)
            if(new_arr != arr):
                accepted += 1
                arr = new_arr.copy()

            if(accepted > 10 * len(arr)):
                break

        distance = calculateSolutionFitness(arr)
        
        
        print("\rTemp: {:.2g} Dist:{:.3f} Accepted:{}".format(T, minDist / scale_factor, accepted), end="\r")
        
        T *= alpha_temp

        if(minDist > distance):
            minDist = distance
            bestRoute = arr.copy()
            eel.update_distance(round(minDist / scale_factor,2), bestRoute)


    e_t = time()
    ex_time = e_t-s_t
    logger.info("CPU execution time: {}".format(ex_time))

    return (calculateSolutionFitness(arr), arr)



def initializeAlgorithm():
    global data, data_type_flag, set_debug, data_cordinate, data_fname, T

    #Controller Variables
    data = CONFIG['DATASET']['FILE_NAME']
    data_type_flag = CONFIG.getint('DATASET', 'DATASET_TYPE')
    set_debug = CONFIG.getboolean('DEBUG', 'LOG_FILE')
    data_cordinate = CONFIG.getboolean('DATASET','CONTAINS_COORDINATES')
    T =  CONFIG.getfloat('SIMULATED ANNEALING','TEMPERATURE')

    data_fname = "./dataset/" + data + ".txt"


@eel.expose 
def runAlgo(temp):

    global T, minDist, bestRoute
    T = temp
    
    minDist = math.inf
    bestRoute = []
    
    route = list(range(numberOfCities))
    route.append(route[0])

    if(len(route) == numberOfCities+1):
        logger.info("Initial solution generated successfully")
    else:
        logger.warning("Problem generating initial population")
        sys.exit()


    minDist,bestRoute = SA(route)

    eel.update_distance(round(minDist / scale_factor,2), bestRoute)

    #logger.info("FITNESS CURVE:\n{}".format(fitness_curve))  
    logger.info("MINIMAL DISTANCE={}".format(minDist / scale_factor))
    logger.info("BEST ROUTE FOUND={}".format(bestRoute))
    logger.info("\nAlgorithm Completed Successfully.")
    
    logger.info("All done")


initializeAlgorithm()
logging_setup()
eel.start("index.html", size=(1600,950))