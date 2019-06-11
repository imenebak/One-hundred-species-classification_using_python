import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score


home = "Data/tr1.txt"

def load_data():
    data = pd.read_csv(home, skiprows=0)
    data.species.value_counts()
    species = data.pop('species')
    # Since the labels are textual, so we encode them categorically
    y = transf_target(species)
    X = data.iloc[:, 1:]
    return X, y, species
    
def split_data():
    x, y, s = load_data()
    #data split
    x_train, x_test, y_train, y_test = train_test_split(x, y,train_size = 1-0.2, test_size = 0.2, random_state =42)
    return (x_train, y_train), (x_test, y_test), x, y, s

def transf_target(Y):
    #Encode labels with value between 0 and nombre_classes-1
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # convert integers to dummy variables 
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y


def baseline_model():
	# create model, une pile linéaire de couches
	model = Sequential()
	
	# ajouter des couches
	model.add(Dense(192, input_dim=192, activation='relu'))
	model.add(Dense(99, activation='softmax'))
	
	'''Instead of sigmoids, most recent deep learning networks use rectified linear units (ReLUs) for the
	hidden layers. A rectified linear unit has output 0 if the input is less than 0, and raw output otherwise.
	That is, if the input is greater than 0, the output is equal to the input.
	ReLUs' machinery is more like a real neuron in your body.'''
	
	'''Les activations ReLU sont évidemment la fonction d'activation non linéaire la plus simple que vous puissiez utiliser.
	Lorsque vous obtenez une entrée positive,la dérivée est égale à 1,
	il n'y a donc pas d'effet de compression que vous rencontrez sur les erreurs rétropropagées de la fonction sigmoïde'''
	
        
	'''La sortie de la fonction softmax est équivalente à une distribution de probabilité catégorique,
           elle vous indique la probabilité que l'une des classes soit vraie.'''

	# Compile model
	# For a multi-class classification problem
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
    
def ploting_loss(a):
    plt.plot(a.history['loss'])
    plt.plot(a.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def ploting_acc(a):
    plt.plot(a.history['acc'])
    plt.plot(a.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
def main():
    print('Loading the training data...')
    (X_num_tr, y_tr), (X_num_val, y_val), x,y, species = split_data()
    print('Training data loaded!')
    #print(X_num_tr.shape, y_tr.shape)
    #print(y_tr)
    
    #Initialiser le modele 
    model = baseline_model()
    
    estimator = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
    
    '''Early stopping : arrêter l'apprentissage lorsque votre perte commence à augmenter (ou, en d'autres termes, la précision de la validation commence à diminuer).
    Monitor val loss : Surveillez la perte de validation
    min_delta : un seuil permettant de quantifier ou non une perte à une époque donnée
    patience : argument représente le nombre d'époques avant d'arrêter une fois que votre perte commence à augmenter '''

    a = model.fit(x,y,validation_data=(X_num_val,y_val),callbacks=[estimator],verbose=2 ,epochs=1000)
    #a = model.fit(x,y,validation_data=(X_num_val,y_val),verbose=0,epochs=100)

    np.set_printoptions(suppress=True)

    ''' suppress=True : imprimez toujours les nombres à virgule flottante en utilisant la notation en virgule fixe.
    Dans ce cas, les nombres égaux à zéro dans la précision actuelle seront imprimés à zéro'''

    #Predection
    pred = model.predict(X_num_val)
    print(pred[0:10])
    print(y_val[0:10])


    # Using the predictions (pred) and the known 1-hot encodings (y_test) we can compute the log-loss error.  
    # The lower a log loss the better.  The probabilities (pred) from the previous section specify how sure the neural network
    # is of its prediction.  Log loss error pubishes the neural network (with a lower score) for very confident, but wrong,
    # classifications.
    print("log-loss error : ",log_loss(y_val,pred))

    # Usually the column (pred) with the highest prediction is considered to be the prediction of the neural network.  It is easy
    # to convert the predictions to the expected iris species.  The argmax function finds the index of the maximum prediction
    # for each row.

    predict_classes = np.argmax(pred,axis=1)
    expected_classes = np.argmax(y_val,axis=1)

    print("Predictions: {}".format(predict_classes))
    print("Expected: {}".format(expected_classes))

    print(species[predict_classes[1:10]])

    # Accuracy might be a more easily understood error metric.  It is essentially a test score.  For all of the iris predictions,
    # what percent were correct?  The downside is it does not consider how confident the neural network was in each prediction.

    correct = accuracy_score(expected_classes,predict_classes)
    print("Accuracy: {}".format(correct))
    ploting_acc(a)
    # loss
    ploting_loss(a)
#main()
