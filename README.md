
# Helemet and Number plate detection using Machine Learning

This project implements a machine learning model for detecting helmets and number plates in images. It is aimed at enhancing safety and compliance on roads by automatically detecting whether motorcyclists are wearing helmets and whether vehicles have visible number plates.


## Installation
To run this project, you need to have Python installed. Clone this repository and install the required dependencies using the following command:

pip install -r requirements.txt
## Usage
After installing the dependencies, you can use the provided scripts to train the model, evaluate its performance, and run inference on new images. Refer to the Model Training and Evaluation sections for more details.

python train.py


## Dataset

The dataset used for training and evaluation consists of annotated images containing motorcycles with and without helmets, as well as vehicles with and without visible number plates. Due to some restrictions, we cannot provide the dataset directly. However, it can be obtained from https://www.shutterstock.com/search/motorcycle-india-helmet.

https://www.alamy.com/stock-photo/without-helmet.html

## Model Training
The model was trained using a convolutional neural network architecture implemented in PyTorch. The training script ('train.py') preprocesses the dataset, constructs the model, and trains it using a specified configuration. Hyperparameters such as learning rate, batch.

The model was already created by training the 'train.py' and the model is 'model2-002.h5'.
## Results
The trained model achieves an accuracy of X% on the test dataset for helmet detection and Y% for number plate detection. Sample results and visualizations are provided in the results directory.