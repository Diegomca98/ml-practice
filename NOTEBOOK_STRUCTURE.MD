# Notebook Structure

## Techniques & Tools
[![README.md](https://img.shields.io/badge/📝-README-lightgrey)](https://github.com/Diegomca98/research-and-learning/blob/dog-breed-id/README.md)
[![Useful Resources](https://img.shields.io/badge/🛠️-Useful_Resources-pink)](https://github.com/Diegomca98/research-and-learning/blob/dog-breed-id/USEFUL_RESOURCES.md)
[![Notes](https://img.shields.io/badge/📝-Notes-lightgrey)](https://github.com/Diegomca98/research-and-learning/blob/dog-breed-id/NOTES.md)

## Step-by-Step
```
 ______________________________________________________________________
|                                                                      |
|                         STEP-BY-STEP PROCESS                         |
|______________________________________________________________________|
|
├── 1. Setup Workspace
|   └── ✔️ Create Google Colab Notebook
|
├── 2. Link Drive
|   └── ✔️ Upload Data to Drive
|
├── 3. Import
|   ├── ✔️ TensorFlow
|   ├── ✔️ TensorFlow Hub
|   └── ✔️ Setting up a GPU for use
|
├── 4. Loading and Checking Data
|   ├── ✔️ Loading Data Labels
|   └── ✔️ Preparing the Images
|
├── 5. Split Data into Sets
|   ├── ✔️ Turning Data Labels into Numbers
|   ├── ✔️ from sklearn.model_selection import train_test_split
|   └── ✔️ Separate Train Data into x_train, y_train, x_val, y_val
|
├── 6. Preprocess Images
|   ├── ✔️ Create Function to Preprocess the Images
|   |   ├── ✔️ Take img_path and img_size as inputs
|   |   ├── ✔️ Use TensorFlow to read the file and save it to a variable called img
|   |   ├── ✔️ Turn our image (jpg) into Tensors
|   |   ├── ✔️ Normalize our image tensor (convert color channel values from 0-255 to 0-1)
|   |   ├── ✔️ Resize the image to be (224,224)
|   |   └── ✔️ Return the modified image
|   |
├── 7. Turn Data into Batches
|   ├── ✔️ Function to return a tuple of Tensors. Takes img_path and label as inputs
|   ├── ✔️ Function to turn our data into 32 sized batches with logic dependant on the type of set (Training, Validation or Test)
|   ├── ✔️ Create and check training and validation data batches
|   └── ✔️ Visualizing Data Batches to better understand the batches
|
├── 8. Preparing Our Inputs and Outputs
|   └── ✔️ Choosing a model that suits our problem using transfer learning
|
├── 9. Model Experiments
|   ├── ✔️ Building the Model
|   |   ├── ✔️ Implementing Callbacks
|   |   |   ├── ✔️ TensorBoard Callback
|   |   |   |   ├── ✔️ Load the TensorBoard Notebook Extension
|   |   |   |   ├── ✔️ Create callback to save logs into a directory
|   |   |   |   ├── ✔️ Pass the callback to our model's fit() function
|   |   |   |   └── ✔️ Visualize our models training logs with the %tensorboard magic function after model training
|   |   |   ├── ✔️ Early Stop Callback
|   |   ├── ✔️ Create function to train the model for experimentation using the previously defined functions and callbacks
|   ├── ✔️ Evaluating the Model
|   └── 🕜 Preventing Overfitting
|
├── 10. Deep Neural Network
|   ├── ❗ Training the DNN
|   ├── ❗ Evaluating Performance with TensorBoard
|   ├── ❗ Make Predictions
|   ├── ❗ Transform Predictions to Text
|   ├── ❗ Visualizing Predictions
|   └── ❗ Evaluate Predictions
|
├── 11. Model
|   ├── ❗ Save Model
|   └── ❗ Load Model
|
├── 12. Test Data Predictions
|   ├── ❗ Predictions with Test Data
|   └── ❗ Predictions with our own Images
|
└── 13. Submit the model to Kaggle
```