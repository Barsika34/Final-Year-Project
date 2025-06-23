from numpy import mean
from numpy import std
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

#Load Dataset
def new_load_dataset():
    data_file='C:/Users/Bars/Desktop/ESC-10/Features2/allfeat2.npy'
    label_file='C:/Users/Bars/Desktop/ESC-10/Features2/labels2.npy'
    data=np.load(data_file)
    labels=np.load(label_file)
    labels_onehot=to_categorical(labels)
    return(data, labels_onehot)

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(40, 51, 1)))
    model.add(AveragePooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(AveragePooling2D((2, 2)))
    #model.add(Conv2D(128, (3, 3), activation='relu'))
    #model.add(MaxPooling2D((2, 2)))
    #model.add(Conv2D(64, (3,3), activation='relu'))
    #model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    #model.add(Dense(64, activation='relu'))
    #model.add(Dense(200, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
	# compile model
    '''
    change optimizer here
    '''
    #opt = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #print(model.summary())
    return model


# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# define model
		model = define_model()
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=100, batch_size=10, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories
 
# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
		# plot loss
		#plt.figure()
        plt.subplot(2, 1, 1)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='orange', label='test')
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
		# plot accuracy
        plt.subplot(2,1,2)
        plt.title('Classification Accuracy')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.7)
    plt.show()

# summarize model performance
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	plt.boxplot(scores)
	plt.show()
#%%    
# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
    trainX, trainY = new_load_dataset()
    print('Training started...')
	# evaluate model
    scores, histories = evaluate_model(trainX, trainY)
	# learning curves
    summarize_diagnostics(histories)
	# summarize estimated performance
    summarize_performance(scores)
 
# entry point, run the test harness
run_test_harness()
