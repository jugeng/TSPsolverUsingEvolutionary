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
from time import process_time


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
scale_factor = 0.000125


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    # Print iterations progress
  
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r{} {} |{}| {}% {} CURR_MIN_DIST={:.2f}'.format(prefix,iteration, bar, percent, suffix, minDist/scale_factor), end = printEnd )
    # Print New Line on Complete
    if iteration == total: 
        print("\n")

 
def addCity_using_coords():
    
    global numberOfCities, cityCoord,distanceMatrix, s_t, e_t

    s_t = process_time()


    temp_list = []
    try:
        with open(str(data_fname), "r") as f:
            for line in f:
                x, y = line.split()
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
    #print(cityCoord, numberOfCities)
    

def addCity_using_dist():
    
    global distanceMatrix, numberOfCities, s_t, e_t
    s_t = process_time()
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
    e_t = process_time()
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

    e_t = process_time()
    logger.info("CPU took {} to complete data loading and distance matrix building".format(e_t-s_t))
    #print(distanceMatrix)
 

#Graphing
def graphing(): 


    fig = plt.figure(figsize = (15,8))
    ax = fig.add_subplot(1, 1, 1)

    # decreasing time
    ax.set_xlabel('Temperature', fontname="Calibri",fontweight="bold", fontsize=14)
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

    x= []
    y= []

    for i in range(len(fitness_curve)):
        x.append(fitness_curve[i][0])
        y.append(fitness_curve[i][1])


    ax.plot(x,y, color = ("#CC3363"), linewidth = 2)
    #ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xticklabels([])

    fig.text(0.8, 0.82, '[INFO]', color = '#86BBD8')
    fig.text(0.8, 0.78, 'MIN DIST={:.4f}'.format(minDist / scale_factor), color = '#F5F1E3')
    fig.text(0.8, 0.76, 'DATASET={}'.format(data), color = '#F5F1E3')
    fig.text(0.8, 0.74, 'TEMPERATURE={}'.format(CONFIG.getfloat('SIMULATED ANNEALING','TEMPERATURE')), color = '#F5F1E3')
    
    fig.text(0.57, 0.02, "TSP solved using Simulated Annealing Algorithm [Visualizer] {}".format(datetime.now()), color = '#86BBD8')

    logger.info("Fitness Curve generated")

    c=0.95
    x = 0.01
    for i in range(len(fitness_curve)):
        if(c < 0.1):
            c = 0.95
            x = 0.88
        fig.text(x,c,"[{}]{}".format(fitness_curve[i][0],fitness_curve[i][1]), fontsize=8, color = "#FAC9B8" )
        c-=0.02

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
    s_t = process_time()

    while(accepted != 0):
        accepted = 0
        for i in range(100 * len(arr)):
            new_arr = testNeighbor(arr)
            comparison = new_arr == arr
            if(comparison.all() == False):
                accepted += 1
                arr = new_arr

            if(accepted > 10 * len(arr)):
                break

        T *= alpha_temp


        distance = calculateSolutionFitness(arr)
        if(minDist > distance):
            minDist = distance
            bestRoute = arr
            fitness_curve.append([format(T, '.3g'), round(minDist  / scale_factor, 4)])
        print("\rAccepted: {} currDist: {}".format(accepted, round(minDist  / scale_factor, 4) ), end = "\r" )


    e_t = process_time()
    logger.info("CPU execution time: {}".format(e_t-s_t))

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

    rname = "./logs/test_SA_{}.csv".format(data)

    try:
        with open(rname, "r") as f:
            for index, l in enumerate(f):
                pass
    except:
        index = -1

    with open(rname, "a") as f:
        f.write("Test {}, {}, {}, {}\n".format(index+2 ,datetime.now(),T, minDist / scale_factor ))

    logger.info("Test results recorded.")


    graphing()


if __name__ == '__main__':

    initializeAlgorithm()
    logging_setup()


    if data_type_flag == 0:
        addCity_using_coords()

    else:
        addCity_using_dist()


    route = np.arange(numberOfCities)


    if(len(route) == numberOfCities):
        logger.info("Initial solution generated successfully")
    else:
        logger.warning("Problem generating initial population")
        sys.exit()

    minDist,bestRoute = SA(route)
        

    logger.info("MINIMAL DISTANCE={}".format(minDist / scale_factor))
    logger.info("BEST ROUTE FOUND={}".format(bestRoute))
    logger.info("\nAlgorithm Completed Successfully.")
    logger.info("FITNESS CURVE:\n{}".format(fitness_curve))   #Will fail if all generations are exhausted


    if(set_debug == True):
        outputRecord()

    logger.info("All done")
