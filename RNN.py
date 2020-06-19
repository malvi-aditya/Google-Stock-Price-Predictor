#Import Libraries
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dropout,Dense,LSTM

#Import dataset and take column to be predicted (open column)
train_data=pd.read_csv('Google_Stock_Price_Train.csv')
train_set=train_data.iloc[:,1:2].values

#Scale the data
scale=MinMaxScaler(feature_range=(0,1))
scaled_train_set=scale.fit_transform(train_set)

#Data Preprocessing

#New data structure created with 60 timesteps & 1 output
x_train,y_train=[],[]
for i in range(60,1258):
    x_train.append(scaled_train_set[i-60:i,0])
    y_train.append(scaled_train_set[i,0])    
x_train,y_train=np.array(x_train),np.array(y_train)

#Reshape x_train, 1 new dimension added 
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1) )


#Build RNN

#Initialise RNN
model=Sequential()

#Add LSTM layers

#First Layer ,x_train.shape[0] automatically taken so not added below in input shape
model.add( LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1) )) 
model.add(Dropout(0.2))
#Second Layer(LSTM)
model.add( LSTM(units=50,return_sequences=True) ) 
model.add(Dropout(0.2))
#Third Layer(LSTM)
model.add( LSTM(units=50,return_sequences=True) ) 
model.add(Dropout(0.2))
#Fourth Layer(LSTM)
model.add( LSTM(units=50) ) 
model.add(Dropout(0.2))
#Output Layer (fully connected layer so Dense)
model.add(Dense(units=1))


#Compile the RNN
model.compile(optimizer='adam',loss='mean_squared_error')

#Fit the RNN to training data
model.fit(x_train,y_train,epochs=100,batch_size=32)

#Save the model
model.save('Stock_Price_Predict_Model')


#Print and Visualise on test data

#Import test data
test_data=pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price=test_data.iloc[:,1:2].values

#Get Predicted Stock Price

#Concatenate train and test data(as any test data requires 60 previous time steps so some required values will be in train data, test data has just 20 values)
full_data=pd.concat( (train_data['Open'],test_data['Open']), axis= 0 )

#60 previous for each day of January 2017 (.values to create numpy array)
inputs=full_data[len(full_data)-len(test_data)-60:].values
inputs=inputs.reshape(-1,1)
inputs=scale.transform(inputs)

#The data structure to be input in RNN
x_test=[]
for i in range(60,80):
    x_test.append(inputs[i-60:i,0])
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1) )


#predict (returns scales data so inverse transform to get actual values)
predicted_stock_price=model.predict(x_test)
predicted_stock_price=scale.inverse_transform(predicted_stock_price)

#Visualise
plt.plot(real_stock_price,color='red',label='Real Google Stock Price')
plt.plot(predicted_stock_price,color='blue',label='Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


