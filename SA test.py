import math
import random

import numpy as np
import pandas as pd
from scipy.constants import k

numberOfCities = 0
genEvolved = 0
minDist = math.inf
bestRoute = []

currentDist = 0

route = []

T = 0.5
scale_factor = 0.000125
alpha_temp = 0.9


def addCity_using_coords():
    
    global numberOfCities, cityCoord, distanceMatrix

    temp_list = []
    try:
        with open("./dataset/ch150_xy.txt", "r") as f:
            for line in f:
                x, y = line.split()
                temp_list.append([float(x),float(y)])  #Convert to float for accuracy
            cityCoord = pd.DataFrame(temp_list, columns = ["x-coord", "y-coord"])
            cityCoord = cityCoord.mul(scale_factor)       #Initiating pandas dataframe
    
        numberOfCities =  len(cityCoord) 
        if numberOfCities > 0:
            distanceMatrix = pd.DataFrame(columns = np.arange(numberOfCities))
            
            generateDistMatrix()

    except: pass

    
def generateDistMatrix():

    global distanceMatrix
    global numberOfCities

    for i in range(numberOfCities):
        temp_dist = []
        for j in range(numberOfCities):
            a = cityCoord.iloc[i].values 
            b = cityCoord.iloc[j].values   

            distance = np.linalg.norm(a-b)
            temp_dist.append(float(distance))  
         

        distanceMatrix.loc[len(distanceMatrix)] = temp_dist


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
    global T

    accepted = 1

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
    
    return (calculateSolutionFitness(arr), arr)


addCity_using_coords()
print("\nCity Count" ,numberOfCities)

route = np.arange(numberOfCities)


a,b = SA(route)
print(a/scale_factor, b)
