"""
George Dittmar CS545 Machine Learning Winter 2012
****HOW TO RUN****

To run the code you must have python with numpy installed. To run the code type
python Neural_net.py [LEARNING-RATE] [EPOCHS] [TRAINING-FILE] [TEST-FILE] [POS-CLASS] [NEG-CLASS]

This code will then run and output the number of epochs used to train, the training set
accuracy, as well as a simple confusion matrix.

HW 1 Code for character recognition. Sets the last element of the input array
as the bias input for simplicity in my brain.

"""
import csv
import sys
import numpy as np
import random
import sys

# fucntion to calculate the output node activation value of 1 or -1 or possibly 0 if the classifier cant decide
def activation(example,input_weights):
  example = np.array(example)
  dot = np.dot(example,input_weights)

  # perform a dot product of the inputs and the weights then sums
  sum = np. sum(dot)
  if sum == 0:
    return 0
  elif sum > 0:
    return 1
  elif sum < 0:
    return -1

#Function to relearn weights after each example is processed.
def calc_error(output,learning_rate,example,input_weights,target):
  for i in range(0,len(input_weights)):
    delta = learning_rate*(target-output)*example[i]
    input_weights[i] = input_weights[i]+ delta

#Read in the test data set and return a tuple containing the list of test features and the list of the target classes. Bias input is added to end of the array.
def Load_data(data,pos,neg):
  examples = list()
  print "Number of examples", len(examples)
  target = list()
  file = open(data,'rt')
  bias = 1
  data_reader = csv.reader(file,quoting = csv.QUOTE_NONNUMERIC)
  for row in data_reader:
    data_class = row[-1]
    features = row[:-1]
    # append the bias to the input vector
    features.append(bias)

    if data_class == pos:
      examples.append(features)
      target.append(1)
    elif data_class == neg:
      examples.append(features)
      target.append(-1)
  return (examples,target)

#calculates the tp,tn,fp,fn and outputs the confusion matrix for the test set
def Accuracy_Calculation(results,target):
  tp = 0
  tn = 0
  fp = 0
  fn = 0
  for i in range(0,len(target)):
    if results[i] == 1 and target[i] == 1:
      tp += 1
    elif results[i] == 1 and target[i] == -1:
      fp += 1
    elif results[i] == -1 and target[i] == 1:
      fn += 1
    elif results[i] == -1 and target[i] == -1:
      tn += 1

  print
  print "Confusion Matrix:"
  print tp,fn
  print tn,fp

  print
  print "Test Accuracy :", float(tp+tn)/float(tp+tn+fp+fn)

# function performs the testing of the perceptron learning algorithm
def Test_Perceptron(test,input_weights,pos,neg):
  examples,target = Load_data(test,pos,neg)
  results = list()
  for row in examples:
    example_res = activation(row,input_weights)
    results.append(example_res)

  Accuracy_Calculation(results,target)

#Main function of execution for the script
def main(l_rate,epoch,training,test,pos,neg):
  examples,target = Load_data(training,pos,neg)
  print "Examples",len(examples)
  input_weights = list()

  # generate random weights from -1 to 1 for all input features given a random seed of 123456
  random.seed(123456)
  for i in range(0,65):
    input_weights.append(random.uniform(-1,1))

  input_weights = np.array(input_weights)

  count_epoch = 0
  prev_acc = 0

  #Train perceptron till there is no change for 10 rounds or user inputs a number of epochs
  epoch_count_down = 10

  while True:
    total_correct = 0
    wrong = 0
  # process example set
    for i in range(0,len(examples)):
      output = activation(examples[i],input_weights)
      if output == target[i]:
        total_correct += 1
      else:
        wrong = wrong+1
      # relearn input weights
      calc_error( output,l_rate,examples[i],input_weights,target[i])
      epoch_acc =  float(total_correct)/float(total_correct+wrong)

    count_epoch += 1


    # if the epoch accuracy = the prev_acc then begin countdown to stop training
    if prev_acc == epoch_acc:
      epoch_count_down  -= 1
      prev_acc = epoch_acc

    # find the current max accuracy
    if epoch_acc > prev_acc:
      epoch_count_down = 10
      prev_acc = epoch_acc

    # if epoch flag is > 0 then train for a certain number of epochs
    if epoch > 0:
      if count_epoch >= epoch:
        break
    else:
      if epoch_count_down == 0:
        #simple break out of while loop to stop training
        break

  print"Number of epochs to train: ", count_epoch
  print "Training Accuracy: ", prev_acc

  Test_Perceptron(test,input_weights,pos,neg)

if __name__ == '__main__':
  if len(sys.argv) < 6  :
    sys.exit("usage: LEARNING-RATE EPOCHS TRAINING-FILE TEST-FILE POS-CLASS NEG-CLASS ")
  l_rate,epoch,training,test,pos,neg = sys.argv[1:7]
  main(float(l_rate),int(epoch),training,test,int(pos),int(neg))
