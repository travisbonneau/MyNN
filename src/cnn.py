################################################################################
#
# LOGISTICS
#
#    Travis Bonneau
#    tmb170230
#
# FILE
#
#    cnn.py
#
# DESCRIPTION
#
#    MNIST image classification with a CNN written and trained in Python
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
#    1. A summary of my cnn.py code:
#
#           The nn.py file is the program that had the most changes. Some of the classes in this file
#       are basically the same from the nn.py, like the Linear, ReLU and Softmax layers, so I'm not going
#       to discuss those as much since they were previously discussed in the nn.py summary, refer to that 
#       file if needed (also there are comments in the code for each class in this file for reference).
#           Like I mentioned before I am mostly going to focus on the forward pass for the new components
#       which are Conv2D, which deals with Conv2D Filters and Conv2D Bias, another new layer is MaxPool
#       and the Flatten layer. The Conv2D filter layer is one of the more difficult layers that was implemented
#       beacuse we are given an image but we need to convert it to a column matrix where the columns represent 
#       the data where the filter is in the image. There is a common method called im2col which I read up online
#       and I provided multiple sources as to where I got the intuition behind the im2col code. The code is
#       pretty easy to read and is heavily commented. So once we have the column matrix (again with each column 
#       representing data where the filter is sliding over the image), and we just perfrom matrix multiplication 
#       like can be seen in slide 20 of the Calculus Lecture. The next layer is the Bias layer for the Conv2D, 
#       which is very similar to any other bias layer that we saw in nn.py, it just adds constants to the image
#       no multiplication takes place, its just a little different because its on a 3D tensor instead of a 1D 
#       one like in nn.py or a 2D one like in extra.py. The other pretty difficult layer was the MaxPool layer.
#       I implemented this one very similarly to the Conv2D Filter layer, where I converted the image to a matrix
#       (a row matrix this time) and then used the max function to get the max value for each fiter location, and 
#       reshaped to fit the output image size (Note: this is done per channel). The last new layer is the flatten
#       layer which just takes a 3D tensor and converts it to a 1D vector, which feeds into the Linear layers and 
#       eventually the softmax layer which I over in the nn.py file.
#           The code for calculating the error/loss, is the same as what it was in nn.py and extra.py which
#       is Cross Entropy loss and is calculated per element based on the predicted and actual values and then
#       summed together to get the total loss.
#           Like the forward pass I am going to mainly focus on the backward pass for the new components for the
#       cnn. The first layer is the Conv2D Filters layer, which in the forward layer er converted to a 2D tesnor
#       to perform matrix-matrix multiplication so, when we get the error I first convert it back to a matrix form
#       so that we can perform regular back prop for matrix-matrix multipication (see slide 20 of the Calculus
#       lecture). The only difference is the sensitivity of the error w.r.t the input needs to be in a 3D tesnor form
#       to be fed back to whatever component it came from, which I achieve by calling col2im, which is basically just
#       the inverse of im2col, we have a bunch of column features and we need to put them back into the image shape
#       from the input image, following the filter rules provieded such as filter rows and cols, stride, and number
#       of input filters. The Conv2D bias layer has a similar back prop as the regular NN bias layer, since it is
#       element wise addition the gradient is a scalar 1, so the gradient and the sensitivity of the error w.r.t
#       the input is equal to the error passed. The most complicated back prop layer was MaxPool because I need to
#       detemine where the max value was in the input image so only those positions get updated in the sensitivity
#       of the error w.r.t the input. This is performed by getting the im2rows matrix from the forward, finding the 
#       max element position and then creating a vector with all positions being 0 except that position wich takes the
#       error passed in. We do this for each row and the call back_row2im, which takes the vectors and converts them to
#       2D tensors and creates the input error channel based on the rows, and this is performed for each channel. 
#       Finally, the last new layer was Flatten, which is simple, it justs reshapes the error back from a 1D tesnor
#       to a 3D tensor, which is then fed into the Conv2D/MaxPool layers.
#           The weight update isn't very different than whats happening in nn.py, I calculate teh gradients 
#       in the backwards function and then apply it in the step function where the learning rate is passed in.
#       The main difference is that there are more components defined here which have trainable weights but
#       even then the code is the same structure. 
#           Sorry this was so long, there was a lot involved in the cnn model, so I felt like I had a lot to go over. If
#       you want more detailed comments, please look at each layers forward and backwards methods as they are heavily
#       commented.
#
#    2. Accuracy display
#
#       epoch= 0, time=2345.39sec, training_loss= 151694.483, testing_accuracy=41.30
#       epoch= 1, time=2350.55sec, training_loss=  72503.627, testing_accuracy=87.64
#       epoch= 2, time=2367.98sec, training_loss=  19995.600, testing_accuracy=94.20
#       epoch= 3, time=2410.27sec, training_loss=  12702.407, testing_accuracy=95.12
#       Final Test Accuracy: 95.12%
#
#    3. Performance display
#
#       Total Time: 9474.19 seconds
#       Convolutional 2D Filters {
#           Input Size:     (1, 28, 28)
#           Output Size:    (1, 28, 28)
#           Parameter Size: (16, 1, 3, 3)
#           MACs:           112896
#       }
#       Convolutional 2D Bias {
#           Input Size:     (16, 28, 28)
#           Output Size:    (16, 28, 28)
#           Parameter Size: (16, 28, 28)
#           MACs:           0
#       }
#       ReLU {
#           Input Size:     (16, 28, 28)
#           Output Size:    (16, 28, 28)
#           Parameter Size: NONE
#           MACs:           12544
#       }
#       Max Pool {
#           Input Size:     (16, 28, 28)
#           Output Size:    (16, 14, 14)
#           Parameter Size: NONE
#           MACs:           0
#       }
#       Convolutional 2D Filters {
#           Input Size:     (16, 14, 14)
#           Output Size:    (16, 14, 14)
#           Parameter Size: (32, 16, 3, 3)
#           MACs:           903168
#       }
#       Convolutional 2D Bias {
#           Input Size:     (32, 14, 14)
#           Output Size:    (32, 14, 14)
#           Parameter Size: (32, 14, 14)
#           MACs:           0
#       }
#       ReLU {
#           Input Size:     (32, 14, 14)
#           Output Size:    (32, 14, 14)
#           Parameter Size: NONE
#           MACs:           6272
#       }
#       Max Pool {
#           Input Size:     (32, 14, 14)
#           Output Size:    (32, 7, 7)
#           Parameter Size: NONE
#           MACs:           0
#       }
#       Convolutional 2D Filters {
#           Input Size:     (32, 7, 7)
#           Output Size:    (32, 7, 7)
#           Parameter Size: (64, 32, 3, 3)
#           MACs:           903168
#       }
#       Convolutional 2D Bias {
#           Input Size:     (64, 7, 7)
#           Output Size:    (64, 7, 7)
#           Parameter Size: (64, 7, 7)
#           MACs:           0
#       }
#       ReLU {
#           Input Size:     (64, 7, 7)
#           Output Size:    (64, 7, 7)
#           Parameter Size: NONE
#           MACs:           3136
#       }
#       Flatten {
#           Input Size:     (64, 7, 7)
#           Output Size:    (1, 3136)
#           Parameter Size: NONE
#           MACs:           0
#       }
#       Matrix Multiplication {
#           Input Size:     (1, 3136)
#           Output Size:    (1, 100)
#           Parameter Size: (3136, 100)
#           MACs:           313600
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
NUM_EPOCHS = 4

# display
DISPLAY_ROWS   = 8
DISPLAY_COLS   = 4
DISPLAY_COL_IN = 10
DISPLAY_ROW_IN = 25
DISPLAY_NUM    = DISPLAY_ROWS*DISPLAY_COLS

################################################################################
#
# MODEL DEFINITION
#
################################################################################

class MNIST_CNN():

    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(input_channels=1, output_channels=16, filter_size=(3, 3), image_size=(28, 28), activation_function="relu")
        self.max1 = MaxPool(filter_size=(3, 3), stride=2)
        self.conv2 = Conv2D(input_channels=16, output_channels=32, filter_size=(3, 3), image_size=(14, 14), activation_function="relu")
        self.max2 = MaxPool(filter_size=(3, 3), stride=2)
        self.conv3 = Conv2D(input_channels=32, output_channels=64, filter_size=(3, 3), image_size=(7, 7), activation_function="relu")
        self.flatten = Flatten()
        self.linear1 = Linear(in_size=3136, out_size=100, activation_function="relu")
        self.linear2 = Linear(in_size=100, out_size=10)
        self.softmax = SoftmaxCrossEntropy(size=10)
        
    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.max1.forward(x)
        x = self.conv2.forward(x)
        x = self.max2.forward(x)
        x = self.conv3.forward(x)
        x = self.flatten.forward(x)
        x = self.linear1.forward(x)
        x = self.linear2.forward(x)
        x = self.softmax.forward(x)
        return x

    def backward(self, error):
        error = self.softmax.backward(error)
        error = self.linear2.backward(error)
        error = self.linear1.backward(error)
        error = self.flatten.backward(error)
        error = self.conv3.backward(error)
        error = self.max2.backward(error)
        error = self.conv2.backward(error)
        error = self.max1.backward(error)
        error = self.conv1.backward(error)
        return error

    def step(self, learning_rate):
        self.conv1.step(learning_rate)
        self.max1.step(learning_rate)
        self.conv2.step(learning_rate)
        self.max2.step(learning_rate)
        self.conv3.step(learning_rate)
        self.flatten.step(learning_rate)
        self.linear1.step(learning_rate)
        self.linear2.step(learning_rate)
        self.softmax.step(learning_rate)

    def zeroize_gradients(self):
        self.conv1.zeroize_gradients()
        self.max1.zeroize_gradients()
        self.conv2.zeroize_gradients()
        self.max2.zeroize_gradients()
        self.conv3.zeroize_gradients()
        self.flatten.zeroize_gradients()
        self.linear1.zeroize_gradients()
        self.linear2.zeroize_gradients()
        self.softmax.zeroize_gradients()

    def __str__(self):
        ret = ""
        ret += str(self.conv1)
        ret += str(self.max1)
        ret += str(self.conv2)
        ret += str(self.max2)
        ret += str(self.conv3)
        ret += str(self.flatten)
        ret += str(self.linear1)
        ret += str(self.linear2)
        ret += str(self.softmax)
        return ret

################################################################################
#
# CONVOLUTIONAL 2D LAYER DEFINITION
#
################################################################################

class Conv2D():

    def __init__(self, input_channels, output_channels, filter_size, image_size, activation_function=None):
        super().__init__()
        self.filters = Conv2DFilters(input_channels=input_channels, output_channels=output_channels, filter_size=filter_size)
        self.bias = Conv2DBias(input_channels=output_channels, image_shape=image_size)
        self.activation_function = None
        if activation_function == "relu":
            self.activation_function = ReLU(size=image_size)

        self.zeroize_gradients()

    def forward(self, x):
        x = self.filters.forward(x)
        x = self.bias.forward(x)
        if self.activation_function is not None:
            x = self.activation_function.forward(x)
        return x

    def backward(self, error):
        if self.activation_function is not None:
            error = self.activation_function.backward(error)
        error = self.bias.backward(error)
        error = self.filters.backward(error)
        return error

    def step(self, learning_rate):
        self.filters.step(learning_rate=learning_rate)
        self.bias.step(learning_rate=learning_rate)
        if self.activation_function is not None:
            self.activation_function.step(learning_rate=learning_rate)

    def zeroize_gradients(self):
        self.filters.zeroize_gradients()
        self.bias.zeroize_gradients()
        if self.activation_function is not None:
            self.activation_function.zeroize_gradients()

    def __str__(self):
        ret = ""
        ret += str(self.filters)
        ret += str(self.bias)
        if self.activation_function is not None:
            ret += str(self.activation_function)
        return ret

################################################################################
#
# CONVOLUTIONAL 2D FILTER LAYER DEFINITION
#
################################################################################

class Conv2DFilters():

    def __init__(self, input_channels, output_channels, filter_size):
        super().__init__()
        self.num_input_channels = input_channels
        self.num_output_channels = output_channels
        self.filter_rows, self.filter_cols = filter_size
        self.padding = self.filter_rows // 2

        self.filters = np.random.rand(output_channels, input_channels * self.filter_rows * self.filter_cols)
        # Weight initilization is like "He-et-al Weight Initialization" but some changes are 
        # made based on some empirical tests.
        self.filters = self.filters * (np.sqrt(2) / (input_channels + output_channels + self.filter_rows + self.filter_cols))
        self.zeroize_gradients()

    '''
    This method is used to convert the image into columns that represent where filters are so
    that we can use this in matrix multiplication to represent Conv2D

    This was one of the most time consuming parts to search and understand so im including 
    these links for reference: 
     - http://cs231n.stanford.edu/slides/2016/winter1516_lecture11.pdf
     - https://wiseodd.github.io/techblog/2016/07/16/convnet-conv-layer/
     - https://petewarden.com/2015/04/20/why-gemm-is-at-the-heart-of-deep-learning/
     - https://towardsdatascience.com/how-are-convolutions-actually-performed-under-the-hood-226523ce7fbf

    I tried to comment as much as I could to make it clear as to what it is doing.

    NOTE: I did see multiple blogs that showed faster implementations than what I have; however, I did not
          understand what was going on in the code and I didn't want to implement anything I didn't 
          understand. Which is why I used the implementation below, because the pools make it clear as to
          what it is doing.
    '''
    def im2col(self, x):
        # defining readable variables that can be used throughout the rest of the method
        channels = x.shape[0]
        rows = x.shape[1]
        cols = x.shape[2]

        # This array holds each column of the img_as_col, it holds them as rows
        # and then a transpose is perfomed at the end to get columns
        img_cols = []
        # Still loop over the image as you would with filters but this time you 
        # copy the image filter sections over to a matrix
        for r in range(rows - self.filter_rows + 1):
            for c in range(cols - self.filter_cols + 1):
                # Gets the filter section accross all channels
                filter_section = x[:, r:r+self.filter_rows, c:c+self.filter_cols]
                img_cols.append(filter_section.flatten())
        # Convert python array of numpy arrays to a numpy array of numpy arrays
        img_cols = np.array(img_cols)
        # Return the transpose of img_cols because currently the filters are rows but
        # (as the name suggests) we need them to be columns
        return np.transpose(img_cols)

    def forward(self, x):
        self.input = x

        # Shape data for the incoming image
        channels, rows, cols = x.shape

        # Padding of input image to keep same image shape for output image
        x = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)
        # Convert image to column representation of image. Allows for simple matrix
        # multiplication for the filters to get the output image (still need to resize after)
        self.img_as_cols = self.im2col(x)
        # Miltiply the image by the filters, equaivalent to taking the filters and 
        # moving them across the image but it is more memory efficient.
        y = self.filters @ self.img_as_cols

        # Each new channel is now a single array so we need to resize to the original shape
        # which is why we padded it with zeros first
        y = y.reshape(self.num_output_channels, rows, cols)
        return y

    '''
    This is essentially the inverse of im2col which was researched on the sites above and the logic
    came primarily from those, with modifications to fit my implementation of the CNN.
    '''
    def col2im(self, img_as_col):
        ret_img = np.zeros(self.input.shape)

        # defining readable variables that can be used throughout the rest of the method
        rows = ret_img.shape[1]
        cols = ret_img.shape[2]

        index = 0
        # Similar to im2col this scrolls through the rows and cols and essentially just puts what is in the img_as_col columns
        # into the actual image view (it actually adds to what is there since there can be some overlap of filters).
        for r in range(0, rows - self.filter_rows + 1):
            for c in range(0, cols - self.filter_rows + 1):
                # Assign each colums filter view to the values in the current column that is being reshaped back to the real filter
                # shapes (from 1D column to a 3D matrix for the channels, rows and columns). Basically takes the column and changes
                # it back to the 3D sub-matrix of the image for that filter position.
                ret_img[:, r:r+self.filter_rows, c:c+self.filter_cols] += img_as_col[:, index].reshape(self.num_input_channels, self.filter_rows, self.filter_cols)
                index += 1
        return ret_img

    def backward(self, error):
        # You can see in the forward function right before we return y we reshape it to be the image shape, we need to 
        # reverse this logic to get back a matrix we can multiply which is what we are doing here. The -1 means it 
        # can calculate what the value should be based on the total number of parameters and the current shape.
        reshaped_error = error.reshape(self.num_output_channels, -1)

        # Since the data is converted from image to img_as_col, we can perform the filters as a matrix multiplication 
        # and since we do that on the forward pass the backwards pass is updating the weights and output error the same
        # way we would for regular matrix multiplication which can be seen below.
        #
        # Since we had the data as column vectors this can be seen on slide 20 of the Calculus notes. (In previous
        # implementations I used row vectors for the data so I referenced slide 19).
        self.filter_gradient += reshaped_error @ self.img_as_cols.T

        # Again, refer to slide 20 of the Calculus notes, not that filters is transpoed here because it is stored as a
        # row matrix but I need it as a column matrix. 
        input_error_as_col = self.filters.T @ reshaped_error

        # This takes the input error that is represented as a column matrix and converts it back to an image, which we
        # can return since that is the shape of the data that was fed into the forward function.
        input_error = self.col2im(input_error_as_col)
        return input_error

    def step(self, learning_rate):
        self.filters = self.filters - (learning_rate * self.filter_gradient)

    def zeroize_gradients(self):
        self.filter_gradient = np.zeros((self.num_output_channels, self.num_input_channels * self.filter_rows * self.filter_cols))

    def __str__(self):
        ret = "Convolutional 2D Filters {\n"
        ret += "    Input Size:     (" + str(self.input.shape[0]) + ", " + str(self.input.shape[1]) + ", " + str(self.input.shape[2]) + ")\n"
        ret += "    Output Size:    (" + str(self.input.shape[0]) + ", " + str(self.input.shape[1]) + ", " + str(self.input.shape[2]) + ")\n"
        ret += "    Parameter Size: (" + str(self.num_output_channels) + ", " + str(self.num_input_channels) + ", " + str(self.filter_rows) + ", " + str(self.filter_cols) + ")\n"
        ret += "    MACs:           " + str(self.filters.shape[0] * self.filters.shape[1] * self.img_as_cols.shape[1]) + "\n"
        ret += "}\n"
        return ret

################################################################################
#
# CONVOLUTIONAL 2D BIAS LAYER DEFINITION
#
################################################################################

class Conv2DBias():

    def __init__(self, input_channels, image_shape):
        super().__init__()
        self.num_input_channels = input_channels
        self.img_rows, self.img_cols = image_shape

        self.bias = np.random.rand(self.num_input_channels, self.img_rows, self.img_cols) 
        # Weight initilization is like "He-et-al Weight Initialization" but some changes are
        # made based on some empirical tests.
        self.bias = self.bias * (np.sqrt(2) / (input_channels + self.img_rows + self.img_cols))
        self.zeroize_gradients()

    def forward(self, x):
        self.input = x
        x = x + self.bias
        return x

    def backward(self, error):
        self.bias_gradient += error
        input_error = error
        return input_error

    def step(self, learning_rate):
        self.bias = self.bias - (learning_rate * self.bias_gradient)

    def zeroize_gradients(self):
        self.bias_gradient = np.zeros((self.num_input_channels, self.img_rows, self.img_cols))

    def __str__(self):
        ret = "Convolutional 2D Bias {\n"
        ret += "    Input Size:     (" + str(self.input.shape[0]) + ", " + str(self.input.shape[1]) + ", " + str(self.input.shape[2]) + ")\n"
        ret += "    Output Size:    (" + str(self.input.shape[0]) + ", " + str(self.input.shape[1]) + ", " + str(self.input.shape[2]) + ")\n"
        ret += "    Parameter Size: (" + str(self.num_input_channels) + ", " + str(self.img_rows) + ", " + str(self.img_cols)  + ")\n"
        ret += "    MACs:           0\n"
        ret += "}\n"
        return ret

################################################################################
#
# MAX POOL LAYER DEFINITION
#
################################################################################

class MaxPool():
    def __init__(self, filter_size, stride):
        super().__init__()
        self.filter_rows, self.filter_cols = filter_size
        self.stride = stride
        self.padding = self.filter_rows // 2

        self.zeroize_gradients()

    def im2row(self, x):
        # defining readable variables that can be used throughout the rest of the method
        rows = x.shape[0]
        cols = x.shape[1]

        img_row = []
        # Still loop over the image as you would with filters but this time you 
        # copy the image filter sections over to a matrix
        for r in range(0, rows - self.filter_rows + 1, self.stride):
            for c in range(0, cols - self.filter_cols + 1, self.stride):
                # Gets the filter section accross all channels
                filter_section = x[r:r+self.filter_rows, c:c+self.filter_cols]
                img_row.append(filter_section.reshape(self.filter_rows * self.filter_cols))
        # Convert python array of numpy arrays to a numpy array of numpy arrays
        img_row = np.array(img_row)
        return img_row

    def forward(self, x):
        self.input = x

        channels = x.shape[0]
        new_rows = x.shape[1] // self.stride
        new_cols = x.shape[2] // self.stride

        # Padding of input image
        x = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)

        pooled_img_channels = []
        self.row_img_channels = []
        for c in range(channels):
            # Get the filters section of the image as columns in a matrix
            img_as_rows = self.im2row(x[c])
            # Save the image as rows for the back prop
            self.row_img_channels.append(img_as_rows)
            # Calculuate the Max Pool image
            pooled_img = np.max(img_as_rows, axis=1)
            pooled_img_channels.append(pooled_img.reshape(new_rows, new_cols))
        return np.array(pooled_img_channels)

    def back_row2im(self, x):
        # defining readable variables that can be used throughout the rest of the method
        rows = self.input.shape[1]
        cols = self.input.shape[2]
        y = np.zeros((rows, cols))

        index = 0
        # Scroll over the image like what would be seen in typical video expaling convolution,
        # it takes the filter and goes over the image, but for this it is doing it in reverse, 
        # as to crate the image based off of the row matrix passed in.
        for r in range(0, rows - self.filter_rows + 1, self.stride):
            for c in range(0, cols - self.filter_cols + 1, self.stride):
                # Similar to col2im it takes the row value and reshapes it to a 2D matrix to put
                # into the return image, basically un-flattening the data to put it back in the 
                # shape of the filter.
                d_row_as_filter = x[index].reshape(self.filter_rows, self.filter_cols)
                y[r:r+self.filter_rows, c:c+self.filter_cols] += d_row_as_filter
                index += 1
        return y

    def backward(self, error):
        x = self.input

        channels = x.shape[0]
        new_rows = x.shape[1] // self.stride
        new_cols = x.shape[2] // self.stride

        # Padding of input image
        x = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant', constant_values=0)

        d_img_channels = []
        # I do this per channel, I couldnt find a better way to do this, this also makes it more 
        # inefficient which is why it is somewhat slow. 
        for c in range(channels):
            # Get the filters section of the image as rows in a matrix
            img_as_rows = self.row_img_channels[c]
            d_img_as_rows = np.zeros_like(img_as_rows, dtype=float)
            # Calculuate the Max Pool image
            max_indexes = np.argmax(img_as_rows, axis=1)
            # Flatten error so we can index it easily
            flatten_error = error[c].flatten()
            # This sets each rows max value index to the appropriate error value, which will be used to
            # get the input error image when we call back_row2im
            for i in range(len(max_indexes)):
                max_index = max_indexes[i]
                d_img_as_rows[i, max_index] = flatten_error[i]
            # Since we have the row matrix with the appropriate error in the correct indexes, we need to
            # convert the rows to the actual image, which we do by calling back_row2im, (which is very 
            # similar to col2im in the Conv2DFilter section).
            temp_channel = self.back_row2im(d_img_as_rows)
            d_img_channels.append(temp_channel)
        # Take the array of each channel and convert to a numpy array so we can get the input error
        # to be the same shape as the data passed into the forward function.
        return np.array(d_img_channels)

    def step(self, learning_rate):
        pass

    def zeroize_gradients(self):
        pass

    def __str__(self):
        ret = "Max Pool {\n"
        ret += "    Input Size:     (" + str(self.input.shape[0]) + ", " + str(self.input.shape[1]) + ", " + str(self.input.shape[2]) + ")\n"
        ret += "    Output Size:    (" + str(self.input.shape[0]) + ", " + str(self.input.shape[1]//2) + ", " + str(self.input.shape[2]//2) + ")\n"
        ret += "    Parameter Size: NONE\n"
        ret += "    MACs:           0\n"
        ret += "}\n"
        return ret

################################################################################
#
# FLATTEN LAYER DEFINITION
#
################################################################################

class Flatten():

    def __init__(self):
        super().__init__()
        self.zeroize_gradients()

    def forward(self, x):
        self.input = x
        # x.size is the total number of values in the matrix
        return x.reshape(1, x.size)

    def backward(self, error):
        back_shape = self.input.shape
        # Reshape back to what was sent into the layer
        return error.reshape(back_shape)

    def step(self, learning_rate):
        pass

    def zeroize_gradients(self):
        pass

    def __str__(self):
        ret = "Flatten {\n"
        ret += "    Input Size:     (" + str(self.input.shape[0]) + ", " + str(self.input.shape[1]) + ", " + str(self.input.shape[2]) + ")\n"
        ret += "    Output Size:    (1, " + str(self.input.size) + ")\n"
        ret += "    Parameter Size: NONE\n"
        ret += "    MACs:           0\n"
        ret += "}\n"
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
        # Weight initilization is like "He-et-al Weight Initialization" but some changes are
        # made based on some empirical tests.
        self.mat = np.random.rand(in_size, out_size) * (np.sqrt(2) / (in_size + out_size))
        
        self.zeroize_gradients()

    def forward(self, x):
        self.input = x
        
        x = np.matmul(x, self.mat)
        return x

    def backward(self, error):
        # Calculate parameter gradient
        # input is transpose so we need to tranpose it again
        self.mat_gradient += self.input.T @ error

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

        # This add the bias vector to each batch row
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

        # Same vectorization that was perfomed in the other files. Allows you to 
        # perform ReLU per element in the numpy matrix.
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
        ret += "    Input Size:     " + str(self.input.shape) + "\n"
        ret += "    Output Size:    " + str(self.input.shape) + "\n"
        ret += "    Parameter Size: NONE\n"
        ret += "    MACs:           " + str(self.input.size) + "\n"
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
        ret += "    Input Size:     " + str(self.input.shape) + "\n"
        ret += "    Output Size:    " + str(self.input.shape) + "\n"
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
# YOUR CODE GOES HERE
#
################################################################################

model = MNIST_CNN()
loss_func = CrossEntropyLoss()

# Variables for tracking data
accuracy_per_epoch = []
total_time = 0.0

# cycle through the epochs
for epoch in range(NUM_EPOCHS):
    # set the learning rate
    learning_rate = 0.001
    train_loss = 0.0

    # cycle through the training data
    start_time = time.time()
    for i in range(DATA_NUM_TRAIN):
        train_img = train_data[i]
        train_img = train_img / 255.0

        label_vector = np.zeros((DATA_CLASSES))
        label_vector[train_labels[i]] = 1

        # forward pass
        y = model.forward(train_img)
        # calculate loss
        train_loss += loss_func.compute_loss(estimate=y, actual=label_vector)
        # back prop
        model.backward(label_vector)
        # weight update
        model.step(learning_rate=learning_rate)
        # zeroize gradient
        model.zeroize_gradients()

    # cycle through the testing data
    num_correct = 0
    for i in range(DATA_NUM_TEST):
        test_img = test_data[i]
        test_img = test_img / 255.0
        # Get the label for the testing data, will be used to compare against returned value and 
        # is used to calculate accuracy
        true_index = test_labels[i]

        # forward pass
        y = model.forward(test_img)
        # accuracy
        if np.argmax(y) == true_index:
            num_correct += 1
    test_acc = 100 * (num_correct / DATA_NUM_TEST)
    end_time = time.time()
    
    # Update variables to keep track of stats
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
print("Final Test Accuracy: {:.2f}%".format(accuracy_per_epoch[NUM_EPOCHS - 1]))
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
    input_image = test_data[i, :, :, :].reshape((1, DATA_ROWS, DATA_COLS)) / 255.0
    y = model.forward(input_image)
    true_label = np.argmax(y)

    # Draw plot of test results
    ax.append(fig.add_subplot(DISPLAY_ROWS, DISPLAY_COLS, i + 1))
    ax[-1].set_title('True: ' + str(test_labels[i]) + ' xNN: ' + str(true_label))
    plt.imshow(img, cmap='Greys')
plt.show()