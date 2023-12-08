# Switch the TensorFlow version 2.1
!pip install tensorflow==2.14

"""## Import libraries and create noisy data"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf
print (tf.__version__)

# Parameters (y = a*x + b)
a=0.6
b=2

# Create noisy data
x_data = np.linspace(-10, 10, num=100000)
y_data = a * x_data + b + np.random.normal(size=100000)
print('Data created successfully')

"""# Create the model
Create the model with a single linear neuron.
"""

# Create the model
model = keras.Sequential()
model.add(keras.layers.Dense(units = 1, activation = 'linear', input_shape=[1]))
model.compile(loss='mse', optimizer="adam")

# Display the model (only 2 parameters to optimize)
model.summary()

"""## Learn with 5 epochs"""

# Learn
model.fit( x_data, y_data, epochs=5, verbose=1)

"""# Predict and display"""

# Compute the output
y_predicted = model.predict(x_data)

# Display the result
plt.scatter(x_data[::500], y_data[::500])
plt.plot(x_data, y_predicted, 'r', linewidth=4)
plt.grid()
plt.show()

print( model.trainable_variables )