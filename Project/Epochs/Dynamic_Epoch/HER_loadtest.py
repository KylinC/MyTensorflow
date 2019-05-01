#import the data_process package
import numpy as np

# import NN_template
from NN_template import *

wih_path = "../mode_para/7_epochs/wih.npy"
who_path = "../mode_para/7_epochs/who.npy"

#number of input, hidden and output hnodes
input_nodes=784
hidden_nodes=100
output_nodes=10

#learning rate is 0.3
'''
while 0.1, performance = 95.41%
while 0.3, performance = 94.48%
'''
learning_rate=0.05

#create instance of neural network
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

# load data
n.load(wih_path, who_path)

# test the neural network
# load the mnist test data CSV file into a list
test_data_file = open("../mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network

#scorecard for how well the network performs,initially empty
scorecard=[]

#go through all the records in the test dataset
for record in test_data_list:
    #split the record by the "," commas
    all_values=record.split(',')
    #correct answer is the first values
    correct_label=int(all_values[0])
    # print(correct_label,"correct label")
    #scale and shift the inputs
    inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    #query the network
    outputs=n.query(inputs)
    #the index of the highest value correponds to the label
    label=numpy.argmax(outputs)
    # print(label,"network's answer")
    #append correct or incorrect to list
    if(label==correct_label):
        #network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        #network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass
    pass
print(scorecard)

#calculate the performance score,the fraction of correct answer

scorecard_array=numpy.asfarray(scorecard)
print("performance = ",scorecard_array.sum()/scorecard_array.size)
# run the network backwards, given a label, see what image it produces