# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
The problem statement for developing a neural network regression model involves predicting a continuous value output based on a set of input features. In regression tasks, the goal is to learn a mapping from input variables to a continuous target variable. This neuron network model named 'AI' consists of 5 layers (1 input layer, 1 output layer and 3 hidden layers). The first hidden layer has units, second with 3 units, third with 4 units and output layer with 1 unit. All the hidden layers consist of activation function 'relu' Rectified Linear Unit.

The relationship between the input and the output is given by :

Output = Input * 5 + 7 in the dataset and applying 2000 epochs to minimize the error and predict the output.

## Neural Network Model

![image](https://github.com/singaravetrivelsenthilkumar/basic-nn-model/assets/120572270/00b6ae51-fdf4-4792-be3d-f047aaa7b0b3)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Singaravetrivel S
### Register Number: 212222220048

### Dependencies
```python
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
```
### Data From Sheets
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('EX NO 1').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
```
### Data Visualization
```python
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})
df
x=df[['INPUT']].values
y=df[['OUTPUT']].values
```
### Data split and preprocessing
```python
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=33)
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train1=scaler.transform(x_train)
```
### Regressive model
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
AI=Sequential([
    Dense(units=5,activation='relu',input_shape=[1]),
    Dense(units=3,activation='relu'),
    Dense(units=4,activation='relu'),
    Dense(units=1)
])
AI.compile(optimizer='rmsprop',loss='mse')
AI.fit(x_train1,y_train,epochs=2000)
```
### Loss calculation
```
loss_df = pd.DataFrame(AI.history.history)
loss_df.plot()
```
### Evaluate the model
```python
x_test1 = scaler.transform(x_test)
AI.evaluate(x_test1,y_test)
```
### Prediction
```python
x_n1 = [[8]]
x_n1_1 = scaler.transform(x_n1)
AI.predict(x_n1_1)
```
## Dataset Information

![image](https://github.com/singaravetrivelsenthilkumar/basic-nn-model/assets/120572270/3a7fdc19-7258-4ef8-85de-23428ed9a0a8)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/singaravetrivelsenthilkumar/basic-nn-model/assets/120572270/ff1382c8-e863-47e9-a113-8169776b3875)

### Training

![image](https://github.com/singaravetrivelsenthilkumar/basic-nn-model/assets/120572270/181e678d-923d-476a-9ae9-1fc266ee3d41)

### Test Data Root Mean Squared Error
![image](https://github.com/singaravetrivelsenthilkumar/basic-nn-model/assets/120572270/c6ca55b5-33db-4c2d-93f1-e30b53ada550)

### New Sample Data Prediction
![image](https://github.com/singaravetrivelsenthilkumar/basic-nn-model/assets/120572270/838b03d8-5793-4e10-8eac-6c1873601029)

## RESULT
A neural network regression model for the given dataset is developed and the prediction for the given input is obtained.
