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


CONFIG = configparser.ConfigParser()
CONFIG.read('controller.ini')

#Calculators
numberOfCities = 0
alpha_temp = 0.9

loc_multiplier = math.pi / 180

#Result Store
minDist = math.inf
bestRoute = []

#Data-plotting
fitness_curve = []

#Performance Measure
s_t = 0.0
e_t = 0.0
ex_time = 0.0
scale_factor = 0.000125

cityCoord = []

def addCity_using_coords():

    global numberOfCities, cityCoord,distanceMatrix, s_t, e_t

    s_t = time()

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

#Graphing
def graphing():


    fig = plt.figure(figsize = (23,8))
    fig.set_facecolor('#05070A')

    ax = fig.add_subplot(1, 2, 1)

    # decreasing time
    ax.set_xlabel('Temperature', fontname="Calibri",fontweight="bold", fontsize=14)
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
    y= []

    for i in range(len(fitness_curve)):
        y.append(fitness_curve[i])

    ax.set_xticklabels([])  
    ax.plot(x,y, color = ("#72B01D"), linewidth = 2)
     


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
    sol.scatter(m,n, color = "#F79824", marker = "^" )
    sol.scatter(m[0],n[0], color = "#F79824", marker = "s", s = 75)
    
    #plt.subplots_adjust(left=0.15)


    fig.text(0.92, 0.82, '[INFO]', color = '#86BBD8')
    fig.text(0.92, 0.78, 'MIN DIST={:.4f}'.format(minDist / scale_factor), color = '#F5F1E3')
    fig.text(0.92, 0.76, 'DATASET={}'.format(data), color = '#F5F1E3')
    fig.text(0.92, 0.74, 'TEMPERATURE={}'.format(CONFIG.getfloat('SIMULATED ANNEALING','TEMPERATURE')), color = '#F5F1E3')

    fig.text(0.70, 0.02, "TSP solved using Simulated Annealing Algorithm [Visualizer] {}".format(datetime.now()), color = '#86BBD8')

    logger.info("Fitness Curve generated")

    plt.subplots_adjust(left=0.15)


    if(set_debug == True):
        hjhsda = "./logs/output_curve/G_SA_{}_{}_{}.png".format(datetime.now().strftime("%d-%m-%y %H_%M"), data, round(minDist / scale_factor))
        fig.savefig(hjhsda, facecolor=fig.get_facecolor(), edgecolor='none')
    logger.info("Fitness Curve exported to logs\nFile name: {}".format(hjhsda) )
    #plt.show()   #To view graph after generating

#Data Logging
def logging_setup():
    global logger

    logger = logging.getLogger('tsp_sa')
    fname = "./logs/test_log/TSP_SA_" + datetime.now().strftime("%d-%m-%y %H_%M") + "_" + data +"_"+ str(T) + ".log"

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


    logger.info("TSP USING Simulated Annealing Algorithm\nDeveloped by Jugen Gawande")
    logger.info(str(datetime.now()))
    logger.info("\nTEMPURATURE={temp} \nDATASET SELECTED={name}".format(temp =T, name = data))


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

    counter  = 0 
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

        fitness_curve.append(round(minDist /scale_factor, 4))

         
    logger.info("Total Iterations: {}".format(len(fitness_curve)))
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


def outputRecord():
    # with open("./logs/visualize_data.txt", "w") as f:
    #     f.write(" ".join(str(item) for item in fitness_curve))
    #     f.write("\n")

    rname = "./logs/SA.csv"

    try:
        with open(rname, "r") as f:
            for index, l in enumerate(f):
                pass
    except:
        index = -1

    with open(rname, "a") as f:
        f.write("Test {}, {}, {}, {:.2g}, {}, {:.3g}, {}\n".format(index+2 ,datetime.now(), data ,CONFIG.getfloat('SIMULATED ANNEALING','TEMPERATURE'), round ((minDist / scale_factor), 2), T, round (ex_time, 4) ))

    logger.info("Test results recorded.")


    graphing()



if __name__ == '__main__':

    initializeAlgorithm()
    logging_setup()


    if data_type_flag == 0:
        addCity_using_coords()

    else:
        addCity_using_dist()


    route = list(range(numberOfCities))
    #route = nearestNeighbourSolution(numberOfCities) 
    route.append(route[0])

    if(len(route) == numberOfCities+1):
        logger.info("Initial solution generated successfully")
    else:
        logger.warning("Problem generating initial population")
        sys.exit()


    minDist,bestRoute = SA(route)
    

    #logger.info("FITNESS CURVE:\n{}".format(fitness_curve))  
    logger.info("MINIMAL DISTANCE={}".format(minDist / scale_factor))
    logger.info("BEST ROUTE FOUND={}".format(bestRoute))
    logger.info("\nAlgorithm Completed Successfully.")
    

    if(set_debug == True):
        outputRecord()

    logger.info("All done")
