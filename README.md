# stock-analysis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from lstm import *
import time
import tensorflow
#Step 1 Load Data
X_train, y_train, X_test, y_test =load_data('sp500.csv', 50, True)
#Step 2 Build Model
model = Sequential()

model.add(LSTM(
    input_dim=1,
    output_dim=50,
    return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(
    100,
    return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(
    output_dim=1))
model.add(Activation('linear'))

start = time.time()
model.compile(loss='mse', optimizer='rmsprop')
print ('compilation time : ', time.time() - start)
#Step 3 Train the model
model.fit(
    X_train,
    y_train,
    batch_size=512,
    nb_epoch=1,
    validation_split=0.05)
#Step 4 - Plot the predictions!
predictions = predict_sequences_multiple(model, X_test, 50, 50)
plot_results_multiple(predictions, y_test, 50)
pred=[]
for i in range(len(predictions)):
    for j in range(len(predictions[i])):
        pred.append(predictions[i][j])
pred=np.array(pred)
y_test=np.array(y_test[0:400])
mse = (np.sum((pred - y_test)**2))/(int(len(y_test)))
print("MEAN SQUARE ERROR IS :" + str(mse))
