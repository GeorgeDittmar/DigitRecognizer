#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 13:12:08 2012

@author: george
"""

import csv
import argparse
import sys
import knn
"""
Load a CSV file.
"""
def load_data(csv_path):
    rows = list()
    file = open(csv_path,'rt')
    data_reader = csv.reader(file,quoting=csv.QUOTE_NONE)
    header = data_reader.next()
    for row in data_reader:
        rows.append([int(x) for x in row])
        
    return rows
  
def main(training,test,k):
    training_examples = load_data(training)
    test_examples = load_data(test)
    print len(training_examples)
    print training_examples[0]
    print len(test_examples)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='C2 training script to allow training models using different prototype schemes.')
    parser.add_argument('training_csv',help="Filepath to the training csv file.")
    parser.add_argument('test_csv',help="Filepath to the test csv file.")
    parser.add_argument('k_value',help="Value for k closest neighbors to the test example.")
    args = parser.parse_args()
    main(args.training_csv,args.test_csv,args.k_value)