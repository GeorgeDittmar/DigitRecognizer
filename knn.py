# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/george/.spyder2/.temp.py
"""
import math
import numpy as np

"""
Distance metric used to determine how close an input example is to its neighbors.
"""
def euclid_dist(inst,example):
    
    distance = 0
    for i in range(0,len(inst)):
        val = (inst[i]-example[i])
        val = val**2
        distance += val
    distance = math.sqrt(distance)
        
    
def knn(test,examples,k):
    
    distances = list()
    for ex in examples:
        #build a tuple holding the classification label and the distance
        label = ex[0]
        inst = ex[1:]
        print label, len(inst)
        
      
   
    dist_array = np.array(distances)
    
    for i in k:
        max_ind = np.argmax(dist_array)
        max_val = np.max(dist_array)
        

