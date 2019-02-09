import numpy
import matplotlib.pyplot

data_file=open("/mnist_dataset/mnist_train_100.csv","r")
data_list=data_file.readlines()
data_file.close()

matplotlib.pyplot.figure()
for i in range(100):
    matplotlib.pyplot.subplot(10,10,i+1)
    all_values=data_list[i].split(',')
    image_array=numpy.asfarray(all_values[1:]).reshape((28,28))
    matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation=None)
matplotlib.pyplot.show()
