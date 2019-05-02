import numpy
#scipy.special for the sigmoid function expit()
import scipy.special
#using below packages to load your own 28x28 .png pictures

import matplotlib.pyplot as plt
# ensure the plots are inside this notebook, not an external window

# scipy.ndimage for rotating image arrays
import scipy.ndimage

# import NN_template
from NN_template import *

# add path_fragment
wih_path_frag = "../mode_para/trained_data_set_3/wih_epoch_"
who_path_frag = "../mode_para/trained_data_set_3/who_epoch_"
wih_path_origin = "../mode_para/trained_data_set_3/wih_origin.npy"
who_path_origin = "../mode_para/trained_data_set_3/who_origin.npy"

#number of input, hidden and output hnodes
input_nodes=784
hidden_nodes=100
output_nodes=10

#learning rate is 0.05
'''
while 0.1, performance = 95.41%
while 0.3, performance = 94.48%
'''
learning_rate=0.05

#epochs is the number of times the training data set is used for training
'''
while epochs=2, performance=95.05%
while epochs=7, performance=96.84%
'''
epochs=70

# define the degree-scope we want to test
min_degree=3
max_degree=30
scale_degree=3

#create instance of neural network
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#load the mnist training data CSV file into a list
training_data_file=open("../mnist_dataset/mnist_train.csv","r")
training_data_list=training_data_file.readlines()
training_data_file.close()
#training the neural network

# load the mnist test data CSV file into a list
test_data_file = open("../mnist_dataset/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# contain all the degree cahnge epochs-trend
correct_pac=[]


print("training init...\n")

n.save(wih_path_origin,who_path_origin)

# train the neural network
for degree in range(min_degree,max_degree,scale_degree):
    n.load(wih_path_origin,who_path_origin)
    # contain the fixed max-angle epochs-trend
    correct_list = []
    for e in range(1,epochs+1,1):
        # calculate the numpy file name
        wih_path=wih_path_frag+str(e)+"_degree_"+str(degree)+".npy"
        who_path=who_path_frag+str(e)+"_degree_"+str(degree)+".npy"
        # go through all records in the training data set
        print ("degree %d epoch %d start..."%(degree,e)),
        for record in training_data_list:
            # split the record by the ',' commas
            all_values = record.split(',')
            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # create the target output values (all 0.01, except the desired label which is 0.99)
            targets = numpy.zeros(output_nodes) + 0.01
            # all_values[0] is the target label for this record

            targets[int(all_values[0])] = 0.99

            n.train(inputs, targets)

            ## create rotated variations
            # rotated anticlockwise by x degrees
            inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), degree, cval=0.01, order=1, reshape=False)
            n.train(inputs_plusx_img.reshape(784), targets)
            # rotated clockwise by x degrees
            inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -degree, cval=0.01, order=1, reshape=False)
            n.train(inputs_minusx_img.reshape(784), targets)

            # rotated anticlockwise by 10 degrees
            #inputs_plus10_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
            #n.train(inputs_plus10_img.reshape(784), targets)
            # rotated clockwise by 10 degrees
            #inputs_minus10_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
            #n.train(inputs_minus10_img.reshape(784), targets)

        # scorecard for how well the network performs,initially empty
        scorecard = []
        # go through all the records in the test dataset
        for record in test_data_list:
            # split the record by the "," commas
            all_values = record.split(',')
            # correct answer is the first values
            correct_label = int(all_values[0])
            # print(correct_label, "correct label")
            # scale and shift the inputs
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            # query the network
            outputs = n.query(inputs)
            # the index of the highest value correponds to the label
            label = numpy.argmax(outputs)
            # print(label, "network's answer")
            # append correct or incorrect to list
            if (label == correct_label):
                # network's answer matches correct answer, add 1 to scorecard
                scorecard.append(1)
            else:
                # network's answer doesn't match correct answer, add 0 to scorecard
                scorecard.append(0)
                pass
            pass
        # print(scorecard)

        # calculate the performance score,the fraction of correct answer

        scorecard_array = numpy.asfarray(scorecard)
        correct_rate=scorecard_array.sum() / scorecard_array.size
        correct_list.append(correct_rate)
        print("degree %d epoch %d performance = %f" %(degree ,e, correct_rate))
        # run the network backwards, given a label, see what image it produces

        # label to test

        print ("degree %d epoch %d succeed!\n"%(degree,e))
        n.save(wih_path,who_path)
        pass

    print(correct_list)
    correct_pac.append(correct_list)
    pass

# build the x-axis acale
x_axis_list=range(1,epochs+1,1)

# build the color_list
color_list=["#FF0000","#FF00FF","#FFCC00","#660000","#000000","#0000FF","#6600FF","#66FF00","#006633","#99FFFF","#996600","#CCFF33"]

# draw the epoch changing graph
plt.figure(figsize=(20,20),dpi=100)

# to check which correct_list to read
read_point=0
for degree in range(min_degree,max_degree,scale_degree):
    label="degree:"+str(degree)
    plt.plot(x_axis_list,correct_pac[read_point],lw=4,label=label,color=color_list[read_point])
    read_point+=1

_xtick_labels = ["{}".format(i) for i in x_axis_list]
plt.xticks(x_axis_list,_xtick_labels)
# plt.yticks(range())

plt.grid(True)
plt.grid(linestyle='--')

plt.legend(loc="upper left")
plt.savefig("../picture/epoch_trend_DR_7_lr05")
plt.show()







#end