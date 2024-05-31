# MNIST Classification and Clustering Using Autoencoders

This project demonstrates the use of autoencoders for feature extraction from the MNIST dataset and explores two approaches to classification: training a classifier on the encoded features and clustering the encoded features using KMeans.

## Project Structure

- `mnist_autoencoder_classification.py`: The main script to perform autoencoder training, classification, and clustering.
- `README.md`: This documentation file.

## Requirements

- Python 3.x
- TensorFlow
- NumPy
- scikit-learn

## Setup

1. Install the required libraries:
   ```sh
   pip install tensorflow numpy scikit-learn
   ```

2. Run the script:
   ```sh
   python mnist_autoencoder_classification.py
   ```

## Script Explanation

### Load and Preprocess Data

We start by loading the MNIST dataset and normalizing the pixel values to the range [0, 1].
```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
input_shape = x_train.shape[1:]
```

### Define Autoencoder Architecture

We define a simple autoencoder with an encoding layer of 512 neurons and a decoding layer that reconstructs the original input.
```python
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.models import Model

input_img = Input(shape=input_shape)
x = Flatten()(input_img)
encoded = Dense(512, activation='relu')(x)
decoded = Dense(np.prod(input_shape), activation='sigmoid')(encoded)
decoded = Reshape(input_shape)(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```

### Train the Autoencoder

We train the autoencoder using the training data and validate it on the test data.
```python
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
```

### Extract Encoded Features

We create a new model to extract the encoded features from the input images.
```python
encoder = Model(input_img, encoded)
encoded_imgs_train = encoder.predict(x_train)
encoded_imgs_test = encoder.predict(x_test)
```

### Classifier Using Encoded Features

We replace the decoder part with a classifier consisting of dense layers.
```python
encoded_input = Input(shape=(512,))
dense = Dense(64, activation='relu')(encoded_input)
output = Dense(10, activation='softmax')(dense)

classifier = Model(encoded_input, output)
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

#### Train the Classifier

We train the classifier using a subset of the encoded training data (10%).
```python
num_labels = int(0.1 * len(y_train))
indices = np.random.choice(len(y_train), num_labels, replace=False)
x_train_small = encoded_imgs_train[indices]
y_train_small = y_train[indices]

classifier.fit(x_train_small, y_train_small, epochs=50, batch_size=32, validation_data=(encoded_imgs_test, y_test))
```

#### Evaluate the Classifier

We evaluate the classifier on the encoded test data.
```python
score = classifier.evaluate(encoded_imgs_test, y_test)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
```

### Clustering Using KMeans

We perform clustering on the encoded features using KMeans.
```python
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

kmeans = KMeans(n_clusters=10, random_state=0)
clusters = kmeans.fit_predict(encoded_imgs_train)
centroids = kmeans.cluster_centers_
```

#### Assign Labels to Clusters

We assign labels to the test data based on the nearest cluster centroids.
```python
def assign_labels(encoded_data, centroids):
    labels = []
    for data in encoded_data:
        distances = np.linalg.norm(centroids - data, axis=1)
        labels.append(np.argmin(distances))
    return np.array(labels)

test_labels = assign_labels(encoded_imgs_test, centroids)
```

#### Evaluate Clustering

We evaluate the clustering performance by comparing the assigned cluster labels to the true labels.
```python
print(f'Clustering Accuracy: {accuracy_score(y_test, test_labels)}')
```

## Conclusion

This project demonstrates the effectiveness of autoencoders for feature extraction from the MNIST dataset. We showed how these features can be used for classification with a small amount of labeled data and for clustering with KMeans. The results highlight the potential of unsupervised learning methods for reducing the dimensionality and improving the efficiency of downstream tasks.
