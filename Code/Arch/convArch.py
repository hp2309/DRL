import tensorflow as tf
N = 1000
#First Convolutional Layer
input_shape = (N, 3, 64, 64)
#model = tf.keras.sequential()
x = tf.random.normal(input_shape)
y = tf.keras.layers.Conv2D(filters= 32, kernel_size = (4, 4), strides=(2, 2), padding="same", data_format="channels_first", dilation_rate= 1, activation='elu', input_shape=(3, 64, 64), use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros")(x)
#Second Convolutional Layer
#model = tf.keras.sequential()
y2 = tf.keras.layers.Conv2D(filters= 64, kernel_size = (4, 4), strides=(2, 2), padding="same", data_format="channels_first", dilation_rate= 1, activation='elu', input_shape=(3, 64, 64), use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros")(y)
#Third Convolutional Layer
input_shape = (N, 3, 64, 64)
#model = tf.keras.sequential()
y3 = tf.keras.layers.Conv2D(filters= 64, kernel_size = (3, 3), strides=(1, 1), padding="same", data_format="channels_first", dilation_rate= 1, activation='elu', input_shape=(3, 64, 64), use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros")(y2)
#Fourth Convolutional Layer
input_shape = (N, 3, 64, 64)
#model = tf.keras.sequential()
y4 = tf.keras.layers.Conv2D(filters= 32, kernel_size = (3, 3), strides=(1, 1), padding="same", data_format="channels_first", dilation_rate= 1, activation='elu', input_shape=(3, 64, 64), use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros")(y3)
#Upsampling Layer
y5 = tf.keras.layers.UpSampling2D(size = (2, 2), data_format="channels_first")(y4)
#Concatenating Layer
y6 = tf.keras.layers.concatenate([y, y5], axis = 1)
#Second Convolutional Layer
#model = tf.keras.sequential()
y7 = tf.keras.layers.Conv2D(filters= 64, kernel_size = (4, 4), strides=(2, 2), padding="same", data_format="channels_first", dilation_rate= 1, activation='elu', input_shape=(3, 64, 64), use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros")(y6)
