import numpy 
import matplotlib.pyplot as plt 
from pandas import read_csv 
import math 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

# convert an array of values into a dataset matrix 
def create_dataset(dataset, look_back=1): 
    dataX, dataY = [], [] 
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0] 
        dataX.append(a) 
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

# fix random seed for reproducibility 
numpy.random.seed(7)

# load the dataset 
dataframe = read_csv("EURUSD.csv", usecols=[1], engine='python')
dataset = dataframe.values 
dataset = dataset.astype('float32')

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets 
train_size = int(len(dataset) * 0.67) 
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1 
look_back = 1 
trainX, trainY = create_dataset(train, look_back) 
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features] 
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1])) 
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create and fit the LSTM network 
model3 = Sequential() 
model3.add(LSTM(4, input_shape=(1,look_back), return_sequences=True))
model3.add(LSTM(2, input_shape=(1,look_back)))
model3.add(Dense(1))
model3.compile(loss='mean_squared_error', optimizer='adam') 
model3.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)

# make predictions 
trainPredict = model3.predict(trainX) 
testPredict = model3.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict) 
trainY = scaler.inverse_transform([trainY]) 
testPredict = scaler.inverse_transform(testPredict) 
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0])) 
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting 
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting 
testPredictPlot = numpy.empty_like(dataset) 
testPredictPlot[:, :] = numpy.nan 
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions 
plt.plot(scaler.inverse_transform(dataset)) 
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot) 
plt.show()


 # shift test predictions for plotting 
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan 
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(testPredictPlot)
plt.show()
# Next day 
last = 1.13
x_input = numpy.array(last)
maxDataset = max(dataset) 
minDataset = min(dataset)
x_input = x_input / (maxDataset - minDataset)
x_input = x_input.reshape((1, 1, 1)) 
nextPred = model3.predict(x_input) 
nextPred = scaler.inverse_transform(nextPred)

print("Last:", last)
print("Pred:", nextPred[0][0])

percentagePM = (nextPred[0][0] - last) / last 
print(round(percentagePM*100,2), "%")


from numpy import array
from keras.models import Sequential 
from keras.layers import LSTM
from keras.layers import Dense


# split a univariate sequence into samples
def split_sequence(sequence, n_steps): 
    X, y = list(), list() 
    for i in range(len(sequence)):
        # find the end of this pattern 
        end_ix = i + n_steps 
        # check if we are beyond the sequence 
        if end_ix > len(sequence)-1:
            break 
        # gather input and output parts of the pattern 
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x) 
        y.append(seq_y) 
    return array(X), array(y)

# define input sequence 
raw_seq = dataframe.values
# choose a number of time steps 
n_steps = 1

# split into samples
X, y = split_sequence(raw_seq, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features] 
n_features = 1 
X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model 
model = Sequential() 
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features))) 
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# fit model 
model.fit(X, y, epochs=100, verbose=0, batch_size=1)
# demonstrate prediction 
outList = [] 
yhat = dataframe.values[-1]

for i in range(150): 
    x_input = array(yhat)
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0) 
    print(yhat) 
    outList.append(yhat[0])
    
    
plt.plot(raw_seq)
plt.show()

plt.plot(outList)
plt.show()