import numpy
#scipy.special for the sigmoid function expit()
import scipy.special
#using below packages to load your own 28x28 .png pictures
import glob
import imageio

import matplotlib.pyplot
# ensure the plots are inside this notebook, not an external window

#neural network class definition
class neuralNetwork:
    #initialise the neural network
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        #set nummber of nodes in each input,hidden,output layer
        self.inodes=inputnodes
        self.hnodes=hiddennodes
        self.onodes=outputnodes

        #link weight matrices, wih and who
        #weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        #w11 w21
        #w12 w22 etc
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))

        #learning learningrate
        self.lr=learningrate

        #activation function is the sigmoid function
        self.activation_function=lambda x:scipy.special.expit(x)

        pass

    #train the neuralNetwork
    def train(self,inputs_list,targets_list):
        #convert inputs list to 2d arrays
        inputs=numpy.array(inputs_list,ndmin=2).T
        targets=numpy.array(targets_list,ndmin=2).T

        #calculate signals into hidden layer
        hidden_inputs=numpy.dot(self.wih,inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs=self.activation_function(hidden_inputs)

        #calculate signals into final output layer
        final_inputs=numpy.dot(self.who,hidden_outputs)
        #calculate the signals emerging from final output layer
        final_outputs=self.activation_function(final_inputs)

        #output layer error is the (target-actual)
        output_errors=targets-final_outputs
        #hidden layer error is the output_errors,split by weights, recombined at hidden nodes
        hidden_errors=numpy.dot(self.who.T,output_errors)

        #update the weights for the links between the hidden and output layers
        self.who+=self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))
        #update the weights for the links between the input and hidden layers
        self.wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs))
        pass

    #query the neural network
    def query(self,inputs_list):
        #convert inputs list to 2d arrays
        inputs=numpy.array(inputs_list,ndmin=2).T

        #calculate signals into hidden layers
        hidden_inputs=numpy.dot(self.wih,inputs)
        #calculate the signals emerging from hidden layers
        hidden_outputs=self.activation_function(hidden_inputs)

        #calculate signals into final output layers
        final_inputs=numpy.dot(self.who,hidden_outputs)
        #calculate the signals emerging from final output layers
        final_outputs=self.activation_function(final_inputs)

        return final_outputs

#number of input, hidden and output hnodes
input_nodes=784
hidden_nodes=100
output_nodes=10

#learning rate is 0.3
'''
while 0.1, performance = 95.41%
while 0.3, performance = 94.48%
'''
learning_rate=0.1

#create instance of neural network
n=neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)

#load the mnist training data CSV file into a list
training_data_file=open("/mnist_dataset/mnist_train.csv","r")
training_data_list=training_data_file.readlines()
training_data_file.close()

#training the neural network

#epochs is the number of times the training data set is used for training
epochs=7
'''
while epochs=2, performance=95.05%
while epochs=7, performance=96.84%
'''

for e in range(epochs):
    #go through all records in the training data set for record in training data mnist_dataset
    for record in training_data_list:
        #split the records by the "," commas
        all_values=record.split(',')
        #scale and shift the inputs
        inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        #create the target output values (all 0.01, except the desired label which is 0.99)
        targets=numpy.zeros(output_nodes)+0.01
        #all_values[0] is the target label for this records
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)
        pass
    pass
#test the neural network

# our own image test data set
our_own_dataset = []

# load the png image data as test data set
for image_file_name in glob.glob('/mnist_dataset/myowndataset/2828_my_own_?.png'):

    # use the filename to set the correct label
    label = int(image_file_name[-5:-4])

    # load image data from png files into an array
    print ("loading ... ", image_file_name)
    img_array = imageio.imread(image_file_name, as_gray=True)

    # reshape from 28x28 to list of 784 values, invert values
    img_data  = 255.0 - img_array.reshape(784)

    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(numpy.min(img_data))
    print(numpy.max(img_data))

    # append label and image data  to test data set
    record = numpy.append(label,img_data)
    our_own_dataset.append(record)

    pass

# test the neural network with our own images

# record to test
items = len(our_own_dataset)

for item in range(items):
    # plot image
    matplotlib.pyplot.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')

    # correct answer is first value
    correct_label = our_own_dataset[item][0]
    # data is remaining values
    inputs = our_own_dataset[item][1:]

    # query the network
    outputs = n.query(inputs)
    print (outputs)

    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    print("network says ", label)
    # append correct or incorrect to list
    if (label == correct_label):
        print ("match!")
    else:
        print ("no match!")
        pass














#end
