# Project Notes
## Useful Files in this Repository Branch

[![README.MD](https://img.shields.io/badge/📝-Notes-lightgrey)](https://github.com/Diegomca98/research-and-learning/blob/dog-breed-id/README.MD)
[![Step by Step Summary Diagram](https://img.shields.io/badge/🔍-Notebook_Diagram-lightblue)](https://github.com/Diegomca98/research-and-learning/blob/dog-breed-id/NOTEBOOK_STRUCTURE.MD)
[![Useful Resources](https://img.shields.io/badge/🛠️-Useful_Resources-pink)](https://github.com/Diegomca98/research-and-learning/blob/dog-breed-id/USEFUL_RESOURCES.MD)

## Notes
* To work with TensorFlow we have to transform the data into Tensors, the Tensors are pretty similar to Numpy Arrays:
  * If preprocessing images we use `matplotlib.pyplot.imread(filename)` we'll get a `h by width by color_channel` Numpy Array and if we transform the image to tensors using `tensorflow.constant(filename)` we will get the same result as the Numpy Array, but as type `tensorflow.Tensor`
  
  ```python
  np_array = matplotlib.pyplot.imread(image)
  tensor = tensorflow.constant(image)

  np_array[:2], tensor[:2]

  <Output>:
    (array([[[ 89, 137,  89],
         [ 76, 124,  76],
         [ 63, 111,  61],
         ...,
         [ 77, 133,  86],
         [ 76, 134,  86],
         [ 76, 134,  86]],
 
        [[ 72, 119,  75],
         [ 67, 114,  68],
         [ 63, 110,  64],
         ...,
         [ 75, 131,  84],
         [ 74, 132,  84],
         [ 74, 132,  84]]], dtype=uint8),
    <tf.Tensor: shape=(2, 350, 3), dtype=uint8, numpy=
    array([[[ 89, 137,  89],
            [ 76, 124,  76],
            [ 63, 111,  61],
            ...,
            [ 77, 133,  86],
            [ 76, 134,  86],
            [ 76, 134,  86]],
    
            [[ 72, 119,  75],
            [ 67, 114,  68],
            [ 63, 110,  64],
            ...,
            [ 75, 131,  84],
            [ 74, 132,  84],
            [ 74, 132,  84]]], dtype=uint8)>)
  ```
* The model we build at first using `tf.keras.Sequential()` is going to look like this in a diagram:
  ```
  A[Input Layer] --> B[KerasLayer from Hub]
    B --> C[Dense Layer (softmax)]
    C --> D[Output Layer]
  ```
  * This model has:
    * An input layer A that accepts input data of shape input_shape
    * Layer 1 B is a KerasLayer loaded from TensorFlow Hub. This applies pretrained feature extraction.
    * Layer 2 C is a dense layer with softmax activation to output probabilities.
    * An output layer D that makes predictions.

  * The key aspects are:
    * 2 layers - the Hub KerasLayer and a Dense output layer
    * Sequential stack - each layer feeds into the next linearly
    * Input shape defined separately when model is built, not in layers
    * Compile adds loss, optimizer, metrics for training
  * Which Activation and Which Loss:
    * ![Which Activation and Which Loss](./assets/activation_loss.png)