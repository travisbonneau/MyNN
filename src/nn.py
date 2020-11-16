################################################################################
#
# LOGISTICS
#
#    Travis Bonneau
#    tmb170230
#
# FILE
#
#    nn.py
#
# DESCRIPTION
#
#    MNIST image classification with an NN written and trained in Python
#
# INSTRUCTIONS
#
#    1. Go to Google Colaboratory: https://colab.research.google.com/notebooks/welcome.ipynb
#    2. File - New Python 3 notebook
#    3. Cut and paste this file into the cell (feel free to divide into multiple cells)
#    4. Runtime - Run all
#
# NOTES
#
#    1. A summary of my nn.py code:
#
#           I modeled my model design based on PyTorch with some minor changes, but a lot of
#       the method names are the same.
#           I broke the model into multiple classes that make up model components to make it 
#       easier to work with and easier to debug. The first component is the Linear layer
#       which is just a matrix multiplication followed by an addition of a bias The Vector Matrix
#       Multiplication layer handles a matrix multiplication with a vector input, and the layer 
#       is constructed with an input size and output size to create the shape of the matrix. The 
#       second part of the Linear layer is the bias which I called Vector Addition layer, which 
#       just adds a bias vector to the input vector, which was the output vector of the Matrix 
#       Multiplication layer. See slide 38 of the Linear Algebra notes for the overall Linear layer,
#       and slide 19 of the Calculus slides to see the forward for each component. The Linear layer 
#       can also have an optional activation function, which was ReLU in this project. For the 
#       forward pass of the ReLU layer, it is pretty simple, we just apply the ReLU function to each 
#       element of the input vector. The last layer is the Softmax Cross Entropy layer, in this layer 
#       it only performs softmax, the Cross Entropy part comes after, but it is called the Softmax
#       Cross Entropy layer because the backwards pass uses this information. The Softmax layer just 
#       applies the softmax function per element in the input vector, again the output vector shape is 
#       the same as the input vector.
#           The error code is pretty simple for this project, it just uses Cross Entropy, calculated
#       from the output of the Softmax class and the true vector (all zeros except for a one 
#       where the true class is). The cross entropy implementation follows the basic equation for
#       cross entropy, as seen in the slides and online. It sums up the -y_actual * log(y_pred) 
#       for all indexes in the vector to get a total loss for the vector.
#           The backwards path runs the exact opposite direction of the forward pass, each component
#       contains a backwards pass implementation. Almost all backwards path implementations were
#       taken from the slides especially slide 19 for the gradients of the error/input for Matrix
#       Multiplication. For Vector addition it was element wise so the sensitivity of the error
#       w.r.t the input is equal to the error of the output and the sensitivity of the error w.r.t
#       the parameters is also equal to the sensitivity of the error w.r.t the output. ReLU was 
#       simple, its an element wise piecewise function so we take the derivative of each equation 
#       assign the value (0 or 1) based on what the input was and then multiply it with the 
#       sensitivity of the error w.r.t the output.Softmax is a little different, the softmax is 
#       knowing we used CE for the error so sensitivity of the error w.r.t the input is just the
#       model output minus the expected, which we derived in the homework.
#           For the weight update I hade a step function that would update the weights/parameters when
#       it was called, based on what the gradient matrix/vector had. We calculated the gradients in
#       the backwards functions when we calculated the sensitivity of the error w.t.r the parameters
#       for each applicable component and we stored it for when we used step, and then step just 
#       updates the weights using the learning rate passed and the stored gradients.
#           I dont have too much extra, almost all forward and backwards code for the NN part was straight
#       from the cookbook part of the calculus slides, so that was incredibly useful.
#
#    2. Accuracy display
#
#       epoch= 0, time=1739.83sec, training_loss=198474.590, testing_accuracy=24.48
#       epoch= 1, time=1702.34sec, training_loss= 36716.389, testing_accuracy=92.39
#       epoch= 2, time=1200.79sec, training_loss= 16624.392, testing_accuracy=95.34
#       epoch= 3, time=1076.41sec, training_loss=  8659.394, testing_accuracy=96.75
#       epoch= 4, time=1075.10sec, training_loss=  6011.964, testing_accuracy=97.39
#       Final Test Accuracy: 97.39
#
#    3. Performance display
#
#       Total Time: 6794.47 seconds
#       Matrix Multiplication {
#           Input Size:     (1, 784)
#           Output Size:    (1, 1000)
#           Parameter Size: (784, 1000)
#           MACs:           784000
#       }
#       Vector Addition {
#           Input Size:     (1, 1000)
#           Output Size:    (1, 1000)
#           Parameter Size: (1, 1000)
#           MACs:           0
#       }
#       ReLU {
#           Input Size:     (1, 1000)
#           Output Size:    (1, 1000)
#           Parameter Size: NONE
#           MACs:           1000
#       }
#       Matrix Multiplication {
#           Input Size:     (1, 1000)
#           Output Size:    (1, 100)
#           Parameter Size: (1000, 100)
#           MACs:           100000
#       }
#       Vector Addition {
#           Input Size:     (1, 100)
#           Output Size:    (1, 100)
#           Parameter Size: (1, 100)
#           MACs:           0
#       }
#       ReLU {
#           Input Size:     (1, 100)
#           Output Size:    (1, 100)
#           Parameter Size: NONE
#           MACs:           100
#       }
#       Matrix Multiplication {
#           Input Size:     (1, 100)
#           Output Size:    (1, 10)
#           Parameter Size: (100, 10)
#           MACs:           1000
#       }
#       Vector Addition {
#           Input Size:     (1, 10)
#           Output Size:    (1, 10)
#           Parameter Size: (1, 10)
#           MACs:           0
#       }
#       Softmax Cross Entropy {
#           Input Size:     (1, 10)
#           Output Size:    (1, 10)
#           Parameter Size: NONE
#           MACs:           0
#       } 
#
################################################################################

################################################################################
#
# IMPORT
#
################################################################################

import os.path
import urllib.request
import gzip
import math
import time
import numpy             as np
import matplotlib.pyplot as plt

################################################################################
#
# PARAMETERS
#
################################################################################

# data
DATA_NUM_TRAIN         = 60000
DATA_NUM_TEST          = 10000
DATA_CHANNELS          = 1
DATA_ROWS              = 28
DATA_COLS              = 28
DATA_VECTOR_LENGTH     = DATA_ROWS * DATA_COLS * DATA_CHANNELS
DATA_CLASSES           = 10
DATA_URL_TRAIN_DATA    = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
DATA_URL_TRAIN_LABELS  = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
DATA_URL_TEST_DATA     = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
DATA_URL_TEST_LABELS   = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
DATA_FILE_TRAIN_DATA   = 'train_data.gz'
DATA_FILE_TRAIN_LABELS = 'train_labels.gz'
DATA_FILE_TEST_DATA    = 'test_data.gz'
DATA_FILE_TEST_LABELS  = 'test_labels.gz'

# training data
TRAINING_LR_MAX          = 0.01
TRAINING_LR_INIT_SCALE   = 0.01
TRAINING_LR_INIT         = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL_SCALE  = 0.01
TRAINING_LR_FINAL        = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE
TRAINING_LR_INIT_EPOCHS  = 2
TRAINING_LR_FINAL_EPOCHS = 3
NUM_EPOCHS               = TRAINING_LR_INIT_EPOCHS + TRAINING_LR_FINAL_EPOCHS

# display
DISPLAY_ROWS   = 8
DISPLAY_COLS   = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM    = DISPLAY_ROWS * DISPLAY_COLS

################################################################################
#
# MODEL DEFINITION
#
################################################################################

class MNIST_NN():

    def __init__(self, input_size, output_size):
        self.linear1 = Linear(in_size=input_size, out_size=1000, activation_function="relu")
        self.linear2 = Linear(in_size=1000, out_size=100, activation_function="relu")
        self.linear3 = Linear(in_size=100, out_size=output_size)
        self.softmax = SoftmaxCrossEntropy(size=output_size)

    def forward(self, x):
        x = self.linear1.forward(x)
        x = self.linear2.forward(x)
        x = self.linear3.forward(x)
        x = self.softmax.forward(x)
        return x

    def backward(self, error):
        error = self.softmax.backward(error)
        error = self.linear3.backward(error)
        error = self.linear2.backward(error)
        error = self.linear1.backward(error)

    def step(self, learning_rate):
        self.linear1.step(learning_rate=learning_rate)
        self.linear2.step(learning_rate=learning_rate)
        self.linear3.step(learning_rate=learning_rate)
        self.softmax.step(learning_rate=learning_rate)

    def zeroize_gradients(self):
        self.linear1.zeroize_gradients()
        self.linear2.zeroize_gradients()
        self.linear3.zeroize_gradients()
        self.softmax.zeroize_gradients()

    def __str__(self):
        ret = ""
        ret += str(self.linear1)
        ret += str(self.linear2)
        ret += str(self.linear3)
        ret += str(self.softmax)
        return ret

################################################################################
#
# LINEAR LAYER DEFINITION
#
################################################################################

class Linear():

    def __init__(self, in_size, out_size, activation_function=None):
        self.in_size = in_size
        self.out_size = out_size

        self.mat = VectorMatrixMult(in_size=in_size, out_size=out_size)
        self.bias = VectorAddition(size=out_size)
        self.activation_function = None
        if activation_function == "relu":
            self.activation_function = ReLU(size=out_size)

        self.zeroize_gradients()

    def forward(self, x):
        x = self.mat.forward(x)
        x = self.bias.forward(x)
        if self.activation_function is not None:
            x = self.activation_function.forward(x)
        return x

    def backward(self, error):
        if self.activation_function is not None:
            error = self.activation_function.backward(error)
        error = self.bias.backward(error)
        error = self.mat.backward(error)
        return error

    def step(self, learning_rate):
        self.mat.step(learning_rate=learning_rate)
        self.bias.step(learning_rate=learning_rate)
        if self.activation_function is not None:
            self.activation_function.step(learning_rate=learning_rate)

    def zeroize_gradients(self):
        self.mat.zeroize_gradients()
        self.bias.zeroize_gradients()
        if self.activation_function is not None:
            self.activation_function.zeroize_gradients()

    def __str__(self):
        ret = ""
        ret += str(self.mat)
        ret += str(self.bias)
        if self.activation_function is not None:
            ret += str(self.activation_function)
        return ret

################################################################################
#
# MATRIX MULTIPLICATION LAYER DEFINITION
#
################################################################################

class VectorMatrixMult():

    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        # Weight initilization is like "He-et-al Weight Initialization" but some changes are
        # made based on some empirical tests.
        self.mat = np.random.randn(in_size, out_size) * (np.sqrt(2 / (in_size * out_size)))
        
        self.zeroize_gradients()

    def forward(self, x):
        self.input = x
        
        x = x @ self.mat
        return x

    def backward(self, error):
        # Calculate parameter gradient
        # Input is transpose so we need to tranpose it again
        # See slide 19 of the Calculus notes
        self.mat_gradient += np.transpose(self.input) @ error

        # Calculate input gradient
        # See slide 19 of the Calculus notes
        input_error = error @ self.mat.T
        return input_error

    def step(self, learning_rate):
        self.mat = self.mat - (learning_rate * self.mat_gradient)

    def zeroize_gradients(self):
        self.mat_gradient = np.zeros((self.in_size, self.out_size))

    def __str__(self):
        ret = "Matrix Multiplication {\n"
        ret += "    Input Size:     (1, " + str(self.in_size) + ")\n"
        ret += "    Output Size:    (1, " + str(self.out_size) + ")\n"
        ret += "    Parameter Size: (" + str(self.in_size) + ", " + str(self.out_size) + ")\n"
        ret += "    MACs:           " + str(self.in_size * self.out_size) + "\n"
        ret += "}\n"
        return ret

################################################################################
#
# VECTOR ADDITION LAYER DEFINITION
#
################################################################################

class VectorAddition():

    def __init__(self, size):
        self.size = size
        # Weight initilization is like "He-et-al Weight Initialization" but some changes are
        # made based on some empirical tests.
        self.vector = np.random.rand(1, size) * (np.sqrt(2) / size)

        self.zeroize_gradients()

    def forward(self, x):
        self.input = x

        x = x + self.vector
        return x

    def backward(self, error):
        # Calculate parameter gradient
        # See slide 11 of the Calculus notes, it's not
        # element wise, but the vector addition are
        # multiple element wise additions
        self.vector_gradient += error

        # Calculate input gradient
        # See slide 11 of the Calculus notes, it's not
        # element wise, but the vector addition are
        # multiple element wise additions
        input_error = error
        return input_error

    def step(self, learning_rate):
        self.vector = self.vector - (learning_rate * self.vector_gradient)

    def zeroize_gradients(self):
        self.vector_gradient = np.zeros((1, self.size))

    def __str__(self):
        ret = "Vector Addition {\n"
        ret += "    Input Size:     (1, " + str(self.size) + ")\n"
        ret += "    Output Size:    (1, " + str(self.size) + ")\n"
        ret += "    Parameter Size: (1, " + str(self.size) + ")\n"
        ret += "    MACs:           0\n"
        ret += "}\n"
        return ret

################################################################################
#
# RELU LAYER DEFINITION
#
################################################################################

class ReLU():

    def __init__(self, size):
        super().__init__()
        self.size = size

        self.zeroize_gradients()
    
    def __relu(self, x):
        if x <= 0:
            return 0
        return x

    def __relu_grad(self, x):
        if x <= 0:
            return 0
        return 1

    def forward(self, x):
        self.input = x

        # This essentially creates a function that will get
        # called for each value in a numpy matrix. So it 
        # will do element wise ReLU as expected.
        relu = np.vectorize(self.__relu)
        x = relu(x)
        return x

    def backward(self, error):
        relu_grad = np.vectorize(self.__relu_grad)

        input_error = np.multiply(relu_grad(self.input), error)
        return input_error

    def step(self, learning_rate):
        pass

    def zeroize_gradients(self):
        pass

    def __str__(self):
        ret = "ReLU {\n"
        ret += "    Input Size:     (1, " + str(self.size) + ")\n"
        ret += "    Output Size:    (1, " + str(self.size) + ")\n"
        ret += "    Parameter Size: NONE\n"
        ret += "    MACs:           " + str(self.size) + "\n"
        ret += "}\n"
        return ret

################################################################################
#
# SOFTMAX CROSS ENTROPY LAYER DEFINITION
#
################################################################################

class SoftmaxCrossEntropy():

    def __init__(self, size):
        super().__init__()
        self.size = size

        self.zeroize_gradients()

    def forward(self, x):
        self.input = x

        # I subtract the max value to get rid of overflow, this makes it
        # so that the max value in the vector is 0 and all other values
        # are negative, so there is no overflow. This is called stable 
        # softmax which I found online in order to avoid overflow.
        e_x = np.exp(x - np.max(x))
        x = e_x / e_x.sum(axis=1)

        self.softmax_x = x
        return x

    def backward(self, error):
        # Takes the output of the softmax and subtracts the expected value
        # from it.
        input_error = np.subtract(self.softmax_x, error)
        return input_error

    def step(self, learning_rate):
        pass

    def zeroize_gradients(self):
        pass

    def __str__(self):
        ret = "Softmax Cross Entropy {\n"
        ret += "    Input Size:     (1, " + str(self.size) + ")\n"
        ret += "    Output Size:    (1, " + str(self.size) + ")\n"
        ret += "    Parameter Size: NONE\n"
        ret += "    MACs:           0\n"
        ret += "}\n"
        return ret

################################################################################
#
# CROSS ENTROPY LOSS FUNCTION
#
################################################################################

class CrossEntropyLoss():

    def __init__(self):
        super().__init__()

    def compute_loss(self, estimate, actual):
        # The 1e-15 was added because I would sometimes get a 0 for the estimate value and 
        # when useing log_{2} i would get -inf so I added a small offset to remove the 
        # posibility of getting -inf for log_{2}.
        loss = -np.sum(actual * np.log2(estimate + 1e-15))
        return loss

################################################################################
#
# DATA
#
################################################################################

# download
if (os.path.exists(DATA_FILE_TRAIN_DATA)   == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_DATA,   DATA_FILE_TRAIN_DATA)
if (os.path.exists(DATA_FILE_TRAIN_LABELS) == False):
    urllib.request.urlretrieve(DATA_URL_TRAIN_LABELS, DATA_FILE_TRAIN_LABELS)
if (os.path.exists(DATA_FILE_TEST_DATA)    == False):
    urllib.request.urlretrieve(DATA_URL_TEST_DATA,    DATA_FILE_TEST_DATA)
if (os.path.exists(DATA_FILE_TEST_LABELS)  == False):
    urllib.request.urlretrieve(DATA_URL_TEST_LABELS,  DATA_FILE_TEST_LABELS)

# training data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_train_data   = gzip.open(DATA_FILE_TRAIN_DATA, 'r')
file_train_data.read(16)
buffer_train_data = file_train_data.read(DATA_NUM_TRAIN*DATA_ROWS*DATA_COLS)
train_data        = np.frombuffer(buffer_train_data, dtype=np.uint8).astype(np.float32)
train_data        = train_data.reshape(DATA_NUM_TRAIN, 1, DATA_ROWS, DATA_COLS)

# training labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_train_labels   = gzip.open(DATA_FILE_TRAIN_LABELS, 'r')
file_train_labels.read(8)
buffer_train_labels = file_train_labels.read(DATA_NUM_TRAIN)
train_labels        = np.frombuffer(buffer_train_labels, dtype=np.uint8).astype(np.int32)

# testing data
# unzip the file, skip the header, read the rest into a buffer and format to NCHW
file_test_data   = gzip.open(DATA_FILE_TEST_DATA, 'r')
file_test_data.read(16)
buffer_test_data = file_test_data.read(DATA_NUM_TEST*DATA_ROWS*DATA_COLS)
test_data        = np.frombuffer(buffer_test_data, dtype=np.uint8).astype(np.float32)
test_data        = test_data.reshape(DATA_NUM_TEST, 1, DATA_ROWS, DATA_COLS)

# testing labels
# unzip the file, skip the header, read the rest into a buffer and format to a vector
file_test_labels   = gzip.open(DATA_FILE_TEST_LABELS, 'r')
file_test_labels.read(8)
buffer_test_labels = file_test_labels.read(DATA_NUM_TEST)
test_labels        = np.frombuffer(buffer_test_labels, dtype=np.uint8).astype(np.int32)

# debug
# print(train_data.shape)   # (60000, 1, 28, 28)
# print(train_labels.shape) # (60000,)
# print(test_data.shape)    # (10000, 1, 28, 28)
# print(test_labels.shape)  # (10000,)

################################################################################
#
# TRAINING OF THE MODEL
#
################################################################################

# learning rate schedule
# This was taken from the example code that you provided. It is the linear warmup followed by 
# a cosine decay
def lr_schedule(epoch):
    # linear warmup followed by cosine decay
    if epoch < TRAINING_LR_INIT_EPOCHS:
        lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
    else:
        lr = (TRAINING_LR_MAX - TRAINING_LR_FINAL)*max(0.0, math.cos(((float(epoch) - TRAINING_LR_INIT_EPOCHS)/(TRAINING_LR_FINAL_EPOCHS - 1.0))*(math.pi/2.0))) + TRAINING_LR_FINAL
    return lr

# Define the model and loss function
model = MNIST_NN(input_size=DATA_VECTOR_LENGTH, output_size=DATA_CLASSES)
loss_func = CrossEntropyLoss()

# Variables for tracking data
accuracy_per_epoch = []
total_time = 0.0

# cycle through the epochs
for epoch in range(NUM_EPOCHS):
    # set the learning rate
    learning_rate = lr_schedule(epoch)
    train_loss = 0.0

    # cycle through the training data
    start_time = time.time()
    for i in range(DATA_NUM_TRAIN):
        train_img = train_data[i]
        train_img = train_img / 255.0
        true_train_label = np.zeros((1, DATA_CLASSES))
        true_train_label[0, train_labels[i]] = 1

        vectorized_train_img = np.reshape(train_img, (1, DATA_VECTOR_LENGTH))

        # forward pass
        y = model.forward(vectorized_train_img)
        # calculate loss
        train_loss += loss_func.compute_loss(estimate=y, actual=true_train_label)
        # back prop
        model.backward(true_train_label)
        # # weight update
        model.step(learning_rate=learning_rate)
        # # zeroize gradient
        model.zeroize_gradients()

    # cycle through the testing data
    num_correct = 0
    for i in range(DATA_NUM_TEST):
        test_img = test_data[i]
        test_img = test_img / 255.0
        true_index = test_labels[i]
        vectorized_test_img = np.reshape(test_img, (1, DATA_ROWS*DATA_COLS*DATA_CHANNELS))

        # forward pass
        y = model.forward(vectorized_test_img)
        # accuracy
        if np.argmax(y) == true_index:
            num_correct += 1
    test_acc = 100 * (num_correct / DATA_NUM_TEST)
    end_time = time.time()

    # Update varibles to keep track of stats
    accuracy_per_epoch.append(test_acc)
    total_time += (end_time - start_time)

    # per epoch display (epoch, time, training loss, testing accuracy, ...)
    print("epoch={:2d}, time={:7.2f}sec, training_loss={:10.3f}, testing_accuracy={:5.2f}".format(epoch, end_time - start_time, train_loss, test_acc))

################################################################################
#
# DISPLAY
#
################################################################################

# Just to make it more readable
print("")

# accuracy display
# final value
print("Final Test Accuracy: {:.2f}".format(accuracy_per_epoch[NUM_EPOCHS - 1]))
# plot of accuracy vs epoch
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.plot(np.arange(0, NUM_EPOCHS), accuracy_per_epoch, marker='o')

# performance display
# total time
print("Total Time: {:.2f} seconds".format(total_time))
# per layer info (type, input size, output size, parameter size, MACs, ...)
print(model)

# example display
# replace the xNN predicted label with the label predicted by the network
fig = plt.figure(figsize=(DISPLAY_COL_IN, DISPLAY_ROW_IN))
ax  = []
for i in range(DISPLAY_NUM):
    img = test_data[i, :, :, :].reshape((DATA_ROWS, DATA_COLS))

    # Get vector version of image and send it into the model
    vector_img = img.reshape(1, DATA_ROWS * DATA_COLS) / 255.0
    y = model.forward(vector_img)
    true_label = np.argmax(y)

    # Draw plot of test results
    ax.append(fig.add_subplot(DISPLAY_ROWS, DISPLAY_COLS, i + 1))
    ax[-1].set_title('True: ' + str(test_labels[i]) + ' xNN: ' + str(true_label))
    plt.imshow(img, cmap='Greys')
plt.show()