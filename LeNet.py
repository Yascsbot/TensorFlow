import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# LeNet model
model = tf.keras.models.Sequential([
    # convolutional layer 
    tf.keras.layers.Conv2D(6, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1)),
    
    # Second convolutional layer 
    tf.keras.layers.Conv2D(16, kernel_size=(5, 5), padding='same', activation='relu'),
    tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(1, 1)),
    
    # Flatten the output
    tf.keras.layers.Flatten(),
    
    # Fully connected layer
    tf.keras.layers.Dense(120, activation='relu'),
    
    # Fully connected layer
    tf.keras.layers.Dense(84, activation='relu'),

    tf.keras.layers.Dense(10, activation='softmax')
])

predictions = model(x_train[:1]).numpy()
predictions     #returns a vector of logits or log-odds scores, one for each class.

#function converts these logits to probabilities for each class
tf.nn.softmax(predictions).numpy()

# loss function for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()   # gives probabilities close to random (1/10 for each class)


model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
#Model.fit method to adjust your model parameters and minimize the loss
model.fit(x_train, y_train, epochs=5)

#Model.evaluate method checks the model's performance
model.evaluate(x_test,  y_test, verbose=2)