import scipy, pickle
import numpy as np
from scipy import stats
from scipy import signal
import pytest
pytest.importorskip('sklearn')
from sklearn import svm

def setup(fileName):
	Xd = pickle.load(open(fileName, 'rb'))
	snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])  # map applies function to every element in the list. lambda is an anonymous function. some other crazy stuff happens
	X = []
	lbl = []
	for mod in mods:  # for all mod in mods refers to looping through all modulation schemes
	    for snr in snrs:  # for all snr in snrs refers to looping through all signal to noise ratios
		X.append(
		    Xd[(mod, snr)])  # X is an empty array that we want the data to go into, it appends the mod/snr from dataset
		for i in range(Xd[(mod, snr)].shape[0]):  lbl.append(
		    (mod, snr))  # iterates through the number of rows->shape[0] is taking n from (n,m) rows x cols
	X = np.vstack(X)  # vertical stack the data

	featuresArr = np.zeros((len(X), 2), dtype=np.complex)
	# welchArr = np.zeros((len(X), 128), dtype=np.float32)

	np.set_printoptions(threshold=np.nan)
	for index in range(0, len(X)): # Getting features to train SVM
	    if index % 2 == 0:
		continue
	    if lbl[index][0] != "16QAM" and lbl[index][0] != "128QAM":
		continue
	    complexVec = X[index][0] + X[index][1] * 1j
	    featuresArr[index][0] = scipy.stats.kstat(complexVec, 2)
	    featuresArr[index][1] = scipy.stats.kstat(complexVec, 4)
	#    featuresArr[index][0] = scipy.stats.skew(complexVec, 0, True, 'raise')
	#    featuresArr[index][1] = scipy.stats.kurtosis(complexVec, 0, True, True, 'raise')
	#    featuresArr[index][2] = np.std(complexVec)
	#    featuresArr[index][3] = np.std(complexVec)/np.mean(complexVec)
	#    welchArr[0] = scipy.signal.welch(complexVec, 1.0, 'hanning', 128, None, None, 'constant', False)[1]
	clf = svm.SVC()
	modlbls = [modlbl[0] for modlbl in lbl]

	clf.fit(featuresArr, modlbls) #,featuresArr[1],featuresArr[2],featuresArr[3],welchArr
#	for index in range(0, len(X)): # Used to predict/test the trained SVM
#	    if index % 2 != 0:
#		continue
#	    if lbl[index][0] != "16QAM" and lbl[index][0] != "128QAM":
#		continue
#	    print("Predict: ", clf.predict(featuresArr[index]))
#	    print("Actual: ", lbl[index])

def main():
    if len(sys.argv == 1):
	print "No file name given by user to train SVM, using 'dataset.dat'"
	setup("dataset.dat")
    else:
	setup(sys.argv[1])
