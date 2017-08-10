# Import all the things we need ---
#   by setting env variables before Keras import you can set up which backend and which GPU it uses
import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution2D, ZeroPadding2D
import pickle, random, sys, keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gnuradio
import subprocess
import time
import sys
#Code from https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb


def loaddataset(datasetfile):
    # Load the dataset ...
    #  You will need to seperately download or generate this file
    Xd = pickle.load(open(datasetfile,'rb'))
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0]) #map applies function to every element in the list. lambda is an anonymous function. some other crazy stuff happens
    X = []
    lbl = []
    for mod in mods: #for all mod in mods refers to looping through all modulation schemes
        for snr in snrs: #for all snr in snrs refers to looping through all signal to noise ratios
            X.append(Xd[(mod,snr)]) #X is an empty array that we want the data to go into, it appends the mod/snr from dataset
            for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr)) #iterates through the number of rows->shape[0] is taking n from (n,m) rows x cols
    X = np.vstack(X) #vertical stack the data
    return X,lbl,mods,snrs

#Overall the code loads the dataset then puts the 2 float32 arrays into a two dimensional array X
#Then puts the labels (mod,snr) into a two dimensional array lbl


def partitiondataset(dataset,lbl,mods):
    # Partition the data
    #  into training and test sets of the form we can train/test on
    #  while keeping SNR and Mod labels handy for each
    np.random.seed(2016)
    n_examples = dataset.shape[0] #Determines the total number of examples we have available
    n_train = n_examples * 0.5 #Determines how many training set entries we should have, does a 50/50 split
    train_idx = np.random.choice(range(0,n_examples), size=int(n_train), replace=False) #By keeping the indices in an array, we are able to reference the SNR at a future point. Note: mod also available but the one hot matrix also can take care of that
    test_idx = list(set(range(0,n_examples))-set(train_idx)) #any values not in training set should be in test set. Does this by subtracting complete set of indices by the indices in the training set
    X_train = dataset[train_idx] #Uses indices to build training dataset
    X_test =  dataset[test_idx] #Uses indices to build testing dataset
    def to_onehot(yy): #Builds a one hot array, where each element of the array corresponds to an modulation scheme
        yy1 = np.zeros([len(yy), max(yy)+1]) #Build array of zeros with rows of input array length and columns of one greater than the largest value in the array (deals with indexing starting at 0, based on how we initialized the label)
        yy1[np.arange(len(yy)),yy] = 1 #np.arrange length will return integers from 0 to length of yy. Setting yy1[x,yy] =1 will make each row have a 1 in the location specified by the yy array
        return yy1
    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx))) #Builds training label set using one hot encoding
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx))) #^^
    return X_train,X_test,Y_train,Y_test,test_idx

def buildmodel( mods,trainingdata,dr ):


    in_shp = list(trainingdata.shape[1:])  # Defines what should be used as the models desired input shape. We use the shape of the first entry of X_train. This yields [2,128] which is a 2x128 array
    classes = mods #sets classes for classifications, use mods
    #dr = 0.75 # dropout rate (%)
    model = models.Sequential()
    model.add(Reshape([1]+in_shp, input_shape=in_shp)) #[1]+[2,128] results in [1,2,128]
    #First layer of model needs an input_shape defined. We tell it to take inputs of [2,128] (actually [N,2,128]) but we also tell it to reshape inputs to [1,2,128] so the next layer receives [N,1,2,128]
    model.add(ZeroPadding2D((0, 2))) #adds two columns to the left and right sides causing next layer to receive [N,1,2,132], purpose is so convolution doesnt resize
    model.add(Convolution2D(256, (1, 3), padding='valid', activation="relu", name="conv1", kernel_initializer='glorot_uniform')) #conv layer with relu activation
    model.add(Dropout(dr)) #dropout layer
    model.add(ZeroPadding2D((2, 0))) #adds two rows to the top and bottom causing next layer to receive [N,1,6,132] i think
    model.add(Convolution2D(128, (2, 3), padding="valid", activation="relu", name="conv2", kernel_initializer='glorot_uniform')) #conv layer with relu activation
    model.add(Dropout(dr)) #dropout layer
    model.add(Flatten()) #flattens, puts everything into one array. Used to feed into the Dense layer
    model.add(Dense(256, activation="softmax", name="dense1", kernel_initializer="he_normal")) #Dense layer, everything is connected, relu activation (determines proper weights with training) uses number of input samples
    model.add(Dropout(dr)) #dropout layer
    model.add(Dense( len(classes), name="dense2", kernel_initializer="he_normal")) #Dense layer with the number of classes
    model.add(Activation('softmax')) #softmax activation layer
    model.add(Reshape([len(classes)])) #reshapes to a 1-D array of length classes (this contains our feature space)
    model.compile(loss='categorical_crossentropy', optimizer='adam') #performs categorical cross entropy to detemrine loss, optimizes with adam??  https://arxiv.org/abs/1412.6980v8
    model.summary() #prints what the model looks like
    return model


def trainmodel(model,epochs,batchsize,weightfile,x_train,x_test,y_train,y_test):
    # Set up some params
    nb_epoch = epochs     # number of epochs to train on
    batch_size = batchsize   # training batch size
    #filepath = 'weightdata.wts.h5'

    model.fit(x_train,
         y_train,
         batch_size=batch_size,
         epochs=nb_epoch,
         verbose=2,
         validation_data=(x_test, y_test),
         callbacks = [
             keras.callbacks.ModelCheckpoint(weightfile, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
             keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto')
         ])


def evaluatemodel(model,weightfile,x_test,y_test,batchsize,classes,lbl,test_idx,snrs,experimentname):
    # we re-load the best weights once training is finished
    model.load_weights(weightfile)
    # Show simple version of performance
    score = model.evaluate(x_test, y_test, verbose=0, batch_size=batchsize)

    def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    test_Y_hat = model.predict(x_test, batch_size=batchsize)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,x_test.shape[0]):
        j = list(y_test[i,:]).index(1)
        k = int(np.argmax(test_Y_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
    	confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    plot_confusion_matrix(confnorm, labels=classes)
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    acc = 1.0 * cor / (cor + ncor)
    plt.savefig(str(experimentname)+'-'+str(score)+'-'+str(acc)+'.png')

'''
    # Plot confusion matrix
    for snr in snrs:
    
        # extract classes @ SNR
        test_SNRs = map(lambda x: lbl[x][1], test_idx)
        test_X_i = x_test[np.where(np.array(test_SNRs) == snr)]
        test_Y_i = y_test[np.where(np.array(test_SNRs) == snr)]
    
        # estimate classes
        test_Y_i_hat = model.predict(test_X_i)
        conf = np.zeros([len(classes), len(classes)])
        confnorm = np.zeros([len(classes), len(classes)])
        for i in range(0, test_X_i.shape[0]):
            j = list(test_Y_i[i, :]).index(1)
            k = int(np.max(test_Y_i_hat[i, :]))
            conf[j, k] = conf[j, k] + 1
        for i in range(0, len(classes)):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
        plt.figure()
        plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)" % (snr))
        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        acc = 1.0 * cor / (cor + ncor)
        plt.savefig(str(experimentname)+'-'+str(snr)+'-'+str(acc)+'.png')
'''

def runexperiment(datfilepath,weightfilepath,namestring,batches):
    X,lbl,mods,snrs = loaddataset(datfilepath)
    X_train, X_test, Y_train, Y_test,test_idx = partitiondataset(X,lbl,mods)
    model = buildmodel(mods,trainingdata=X_train,dr=.5)
    trainmodel(model=model,epochs=1000,batchsize=batches,weightfile=weightfilepath,x_train=X_train,x_test=X_test,y_train=Y_train,y_test=Y_test)
    evaluatemodel(model=model,weightfile=weightfilepath,x_test=X_test,y_test=Y_test,batchsize=batches,classes=mods,lbl=lbl,test_idx=test_idx,snrs=snrs,experimentname=namestring)


def predictscheme(data,weightfilepath):
    X,lbl,mods,snrs = loaddataset('dataset.dat')
    X_train, X_test, Y_train, Y_test,test_idx = partitiondataset(X,lbl,mods)
    model = buildmodel(mods,trainingdata=X_train,dr=.5)
    model.load_weights(weightfilepath)
    print(mods[np.argmax(model.predict(data))])
    

def main():
    if sys.argv[1].upper() == "PREDICT":
	import USRPsamples
	USRPdata = USRPsamples.main()
	print(USRPdata)
	predictscheme(USRPdata, 'activeweights.wts.h5')
    else: # assuming training
    	runexperiment(datfilepath=sys.argv[1],weightfilepath=sys.argv[2],namestring='accuracyImage',batches=256)

if __name__ == '__main__':
    main()

