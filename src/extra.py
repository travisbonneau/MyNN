################################################################################
#
# LOGISTICS
#
#    Travis Bonneau
#    tmb170230
#
# FILE
#
#    extra.py
#
# DESCRIPTION
#
#    MNIST image classification with an xNN written and trained in Python
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
#    1. A summary of my extra.py code:
#
#           The overall network is very similar to the one in nn.py, but this one can accept batches
#       of input images that were vectorized.
#           The network is made of components like in nn.py. Since we are dealing with batching the 
#       Linear layer needs to to perform matrix multiplication intead of vector matrix multiplication
#       essentially doing vector matrix mult per row of the input matrix (each row is an image). The 
#       bias is basically the same except it adds the bias to each row. Like the bias layer, the ReLU
#       and Softmax layers are applied to each input row independendently. These are the main changes
#       from nn.py for the forward pass. Each layer is separated and more specific information can be
#       seen in the comments within the code.
#           The calculation of the error doesn't really change much since it still does Cross Entropy on 
#       each element using the actualy and predicted values and then sums over the values to get the 
#       total loss, which isn't really any different than nn.py, but just know that it does it in 
#       batches instead of one at a time.
#           Like the forward pass the backwards pass, while similar to nn.py, was updated for batching.
#       For the Linear layer, the matrix multiplication layer is updated changed by using slide 19
#       of the calculus slides, this is the main change because it now uses matrix multiplication
#       vs vector matrix multiplication which makes the gradient of the parameters calculated by a
#       matrix-matrix multiplication vs a vector-vector multiplication like in nn.py. When 
#       calculating the gradients for the bias we need to sum each row of the input error and then
#       average it to get the average gradient (used in weight update). The ReLU layer isn't really
#       all that different, it just performs the same operation on a matrix. The softmax backprop
#       is calculated using each predicted row be subtracted from the predicted row.
#           The weight update code is exactly the same but the gradient is calculated differently in each
#       backwards function. The main difference is we divided by the number of batches to get the
#       average parameter error. For bias we need to sum the rows first and then divide, each row is
#       the gradient for that input image. The actual weight update is the same though since the 
#       parameters structure shapes do not change.
#           I did create some helper classes to help with preprocessing the data and batching the input
#       images, based on whateer transformations needed to be applied. The Batch Generator just needs
#       to take a batch size that evenely divides the total number of images (so if you change the batch
#       size make sure this is met or an error will occur).
#
#    2. Accuracy display
#
#       epoch= 0, time=  78.68sec, training_loss= 265871.390, testing_accuracy=10.32
#       epoch= 1, time=  83.05sec, training_loss= 350580.184, testing_accuracy=63.04
#       epoch= 2, time=  77.25sec, training_loss=  76078.769, testing_accuracy=82.42
#       epoch= 3, time=  86.53sec, training_loss=  42618.389, testing_accuracy=87.18
#       epoch= 4, time=  89.09sec, training_loss=  34831.319, testing_accuracy=89.45
#       epoch= 5, time=  76.58sec, training_loss=  30020.054, testing_accuracy=90.61
#       epoch= 6, time=  91.11sec, training_loss=  27299.757, testing_accuracy=91.43
#       epoch= 7, time=  85.54sec, training_loss=  25347.016, testing_accuracy=91.88
#       epoch= 8, time=  84.88sec, training_loss=  23833.143, testing_accuracy=92.31
#       epoch= 9, time=  96.32sec, training_loss=  22613.857, testing_accuracy=92.52
#       epoch=10, time=  81.82sec, training_loss=  21648.705, testing_accuracy=92.81
#       epoch=11, time=  83.72sec, training_loss=  20850.076, testing_accuracy=93.03
#       epoch=12, time= 110.16sec, training_loss=  20252.703, testing_accuracy=93.11
#       epoch=13, time=  76.54sec, training_loss=  19826.563, testing_accuracy=93.41
#       epoch=14, time=  79.54sec, training_loss=  19569.606, testing_accuracy=93.61
#       Final Test Accuracy: 93.61
#
#    3. Performance display
#
#       Total Time: 1280.82 seconds
#       Matrix Multiplication {
#           Input Size:     (BATCH_SIZE, 784)
#           Output Size:    (BATCH_SIZE, 1000)
#           Parameter Size: (784, 1000)
#           MACs:           BATCH_SIZE x 784 x 1000 = BATCH_SIZE x 784000
#       }
#       Vector Addition {
#           Input Size:     (BATCH_SIZE, 1000)
#           Output Size:    (BATCH_SIZE, 1000)
#           Parameter Size: (1, 1000)
#           MACs:           0
#       }
#       ReLU {
#           Input Size:     (BATCH_SIZE, 1000)
#           Output Size:    (BATCH_SIZE, 1000)
#           Parameter Size: NONE
#           MACs:           BATCH_SIZE x 1000
#       }
#       Matrix Multiplication {
#           Input Size:     (BATCH_SIZE, 1000)
#           Output Size:    (BATCH_SIZE, 100)
#           Parameter Size: (1000, 100)
#           MACs:           BATCH_SIZE x 1000 x 100 = BATCH_SIZE x 100000
#       }
#       Vector Addition {
#           Input Size:     (BATCH_SIZE, 100)
#           Output Size:    (BATCH_SIZE, 100)
#           Parameter Size: (1, 100)
#           MACs:           0
#       }
#       ReLU {
#           Input Size:     (BATCH_SIZE, 100)
#           Output Size:    (BATCH_SIZE, 100)
#           Parameter Size: NONE
#           MACs:           BATCH_SIZE x 100
#       }
#       Matrix Multiplication {
#           Input Size:     (BATCH_SIZE, 100)
#           Output Size:    (BATCH_SIZE, 10)
#           Parameter Size: (100, 10)
#           MACs:           BATCH_SIZE x 100 x 10 = BATCH_SIZE x 1000
#       }
#       Vector Addition {
#           Input Size:     (BATCH_SIZE, 10)
#           Output Size:    (BATCH_SIZE, 10)
#           Parameter Size: (1, 10)
#           MACs:           0
#       }
#       Softmax Cross Entropy {
#           Input Size:     (BATCH_SIZE, 10)
#           Output Size:    (BATCH_SIZE, 10)
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
BATCH_SIZE               = 30 # The batch size needs to be an even divisor of the training data
TRAINING_LR_MAX          = 0.01
TRAINING_LR_INIT_SCALE   = 0.01
TRAINING_LR_INIT         = TRAINING_LR_MAX*TRAINING_LR_INIT_SCALE
TRAINING_LR_FINAL_SCALE  = 0.01
TRAINING_LR_FINAL        = TRAINING_LR_MAX*TRAINING_LR_FINAL_SCALE
TRAINING_LR_INIT_EPOCHS  = 4
TRAINING_LR_FINAL_EPOCHS = 11
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

        self.mat = MatrixMult(in_size=in_size, out_size=out_size)
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

class MatrixMult():

    def __init__(self, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        # Weight initilization is "He-et-al Weight Initialization" which I read was a
        # good weight initilization method.
        self.mat = np.random.rand(in_size, out_size) * np.sqrt(2 / (in_size + out_size))
        
        self.zeroize_gradients()

    def forward(self, x):
        self.input = x
        
        x = np.matmul(x, self.mat)
        return x

    def backward(self, error):
        batch_size = error.shape[0]

        # Calculate parameter gradient
        # input is transpose so we need to tranpose it again
        # See slide 19 of the Calculus notes
        #
        # Since input is already a matrix then the transpose 
        # of input is a matrix too and will get summed together 
        # due to the matrix multiplication (something which took
        # me a while to realize before I did the math by hand).
        self.mat_gradient += self.input.T @ error

        # Divide by batch size to average the gradient for all batches sent in
        self.mat_gradient /= batch_size

        # Calculate input gradient
        # See page 19 of Calculus Notes
        input_error = error @ self.mat.T
        return input_error

    def step(self, learning_rate):
        self.mat = self.mat - (learning_rate * self.mat_gradient)

    def zeroize_gradients(self):
        self.mat_gradient = np.zeros((self.in_size, self.out_size))

    def __str__(self):
        ret = "Matrix Multiplication {\n"
        ret += "    Input Size:     (BATCH_SIZE, " + str(self.in_size) + ")\n"
        ret += "    Output Size:    (BATCH_SIZE, " + str(self.out_size) + ")\n"
        ret += "    Parameter Size: (" + str(self.in_size) + ", " + str(self.out_size) + ")\n"
        ret += "    MACs:           BATCH_SIZE x " + str(self.in_size) + " x " + str(self.out_size) + " = BATCH_SIZE x " + str(self.in_size * self.out_size) + "\n"
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
        # Weight initilization is "He-et-al Weight Initialization" which I read was a
        # good weight initilization method.
        self.vector = np.random.rand(1, size) * np.sqrt(2 / size)

        self.zeroize_gradients()

    def forward(self, x):
        self.input = x

        # This add the bias vector to each batch row
        x = x + self.vector
        return x

    def backward(self, error):
        # Calculate parameter gradient
        batch_size = error.shape[0]
        # We want to sum all the rows together and we will divide them by 
        # the batch size to get the average gradient value.
        batch_error = np.sum(error, axis=0)

        # Divide by batch size to average the gradient for all batches sent in
        self.vector_gradient += (batch_error / batch_size)

        # Calculate input gradient
        input_error = error
        return input_error

    def step(self, learning_rate):
        self.vector = self.vector - (learning_rate * self.vector_gradient)

    def zeroize_gradients(self):
        self.vector_gradient = np.zeros((1, self.size))

    def __str__(self):
        ret = "Vector Addition {\n"
        ret += "    Input Size:     (BATCH_SIZE, " + str(self.size) + ")\n"
        ret += "    Output Size:    (BATCH_SIZE, " + str(self.size) + ")\n"
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
        ret += "    Input Size:     (BATCH_SIZE, " + str(self.size) + ")\n"
        ret += "    Output Size:    (BATCH_SIZE, " + str(self.size) + ")\n"
        ret += "    Parameter Size: NONE\n"
        ret += "    MACs:           BATCH_SIZE x " + str(self.size) + "\n"
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

    def __softmax(self, x):
        e_x = np.exp(x - np.max(x))
        x = e_x / e_x.sum()
        return x

    def forward(self, x):
        self.input = x

        # This applies softmax to each row in the matrix (each row is one training data)
        # The number of rows is equal to the size of the batch
        # For each row it calls __softmax which returns a row with the softmax function applied
        x = np.apply_along_axis(self.__softmax, 1, x)

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
        ret += "    Input Size:     (BATCH_SIZE, " + str(self.size) + ")\n"
        ret += "    Output Size:    (BATCH_SIZE, " + str(self.size) + ")\n"
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
        # posibility of getting -inf for log_{2}
        loss = -np.sum(actual * np.log2(estimate + 1e-15))
        return loss

################################################################################
#
# TRANSFORMER
#
################################################################################

class Transformer():
    
    def __init__(self, transforms_list):
        super().__init__()
        self.transforms_list = transforms_list

    def apply_transformations(self, data):
        for transformation in self.transforms_list:
            data = transformation.apply(data)
        return data

################################################################################
#
# SCALE TRANSFORMATION
#
################################################################################

class Scale():
    
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def apply(self, data):
        return self.scale * data

################################################################################
#
# VECTORIZE TRANSFORMATION
#
################################################################################

class Vectorize():

    def __init__(self):
        super().__init__()

    def apply(self, data):
        num_data = data.shape[0]
        flatten = 1
        for i in data.shape:
            flatten = flatten * i
        flatten = flatten // num_data

        data = data.reshape((num_data, flatten))
        return data

################################################################################
#
# BATCH GENERATOR
#
################################################################################

class BatchGenerator():

    def __init__(self, batch_size, data):
        super().__init__()
        num_data = data.shape[0]
        
        temp = np.split(data, num_data // batch_size)
        self.batches = np.array(temp)

    def get_batches(self):
        return self.batches

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
def lr_schedule(epoch):
    # linear warmup followed by cosine decay
    if epoch < TRAINING_LR_INIT_EPOCHS:
        lr = (TRAINING_LR_MAX - TRAINING_LR_INIT)*(float(epoch)/TRAINING_LR_INIT_EPOCHS) + TRAINING_LR_INIT
    else:
        lr = (TRAINING_LR_MAX - TRAINING_LR_FINAL)*max(0.0, math.cos(((float(epoch) - TRAINING_LR_INIT_EPOCHS)/(TRAINING_LR_FINAL_EPOCHS - 1.0))*(math.pi/2.0))) + TRAINING_LR_FINAL
    return lr

# Transformer will scale the image and vectorize the 2D image to a vector
transformer = Transformer( [Scale(scale=1/255.0), Vectorize()] )
# Apply the transforms to the training and testing data
trans_train_data = transformer.apply_transformations(data=train_data)
trans_test_data = transformer.apply_transformations(data=test_data)

# Batch Generator will create batches of defined size to feed into the network
# I found a batch size of 30 to produce good results
train_batch_generator = BatchGenerator(batch_size=BATCH_SIZE, data=trans_train_data)

# Initialization of the Model and Loss Function
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
    index = 0
    start_time = time.time()
    for train_img in train_batch_generator.get_batches():
        label_batch = []
        for i in range(BATCH_SIZE):
            label_vector = np.zeros((DATA_CLASSES))
            label_vector[train_labels[index]] = 1
            label_batch.append(label_vector)
            index += 1
        label_batch = np.array(label_batch)

        # forward pass
        y = model.forward(train_img)
        # calculate loss
        train_loss += loss_func.compute_loss(estimate=y, actual=label_batch)
        # back prop
        model.backward(label_batch)
        # weight update
        model.step(learning_rate=learning_rate)
        # zeroize gradient
        model.zeroize_gradients()

    # cycle through the testing data
    num_correct = 0
    index = 0
    for test_img in trans_test_data:
        # Get the label for the testing data, will be used to compare against returned value and 
        # is used to calculate accuracy
        true_index = test_labels[index]
        index += 1

        # forward pass
        y = model.forward(test_img)
        # accuracy
        if np.argmax(y) == true_index:
            num_correct += 1
    test_acc = 100 * (num_correct / DATA_NUM_TEST)
    end_time = time.time()

    # Update varibles to keep track of stats
    accuracy_per_epoch.append(test_acc)
    total_time += (end_time - start_time)

    # per epoch display (epoch, time, training loss, testing accuracy, ...)
    print("epoch={:2d}, time={:7.2f}sec, training_loss={:11.3f}, testing_accuracy={:5.2f}".format(epoch, (end_time - start_time), train_loss, test_acc))

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