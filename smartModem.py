import subprocess
import sys


def chooseFunc():
    program = raw_input("Type the corresponding character for your desired function: \n(D)ataset generation, (C)lassifier training, (T)ransmit data, (R)eceive data\n")
    if program != "":
        if program[0].upper() == "D":
            dataGen()
        elif program[0].upper() == "C":
	    trainCNN()
        elif program[0].upper() == "T":
	    transmitData()
        elif program[0].upper() == "R":
            receiveData()
	else:
            print("Error: Invalid response given. Please type: D, C, T, R to use the corresponding function.")
            chooseFunc()
    else:
        chooseFunc()

def dataGen():
    mods = raw_input("What schemes to use? For all schemes, simply type all. For a list of the schemes, type 'list'. Otherwise, type the schemes with comma separation.\n")
    if mods.upper() == "LIST":
	print("Avaliable schemes are: FM,AM-SSB,AM-DSBFC,AM-DSBSC,16QAM,32QAM,64QAM,128QAM,256QAM,QPSK,BPSK,NOISE")
	dataGen()
    snrs = raw_input("What SNRs to use? For all SNRs (-20,-10,0,10,20), simply type all. Otherwise, type the SNRs with comma separation.\n")
    noiseFile = "noise.wav"
    datasetFile = raw_input("Name of dataset file to save to?\n")
    print "Attempting to generate dataset...\n"
    paramString = "python datasetgen.py "+mods+" "+snrs+" "+noiseFile+" "+datasetFile
    returnValue = subprocess.call(paramString, shell=True)
    if returnValue == 0:
        print("Datasetgen was run successfully.")
    else:
        print("Datasetgen encountered an error. Return value: {}\n".format(returnValue))
	print("Please check the parameters that were used. datasetgen.py was called with: {}\n".format(paramString))

def trainCNN():
    setName = raw_input("Path to dataset file to train with? Press enter to default to dataset.dat\n")
    if setName == "":
        print "Received no input, initializing dataset as dataset.dat."
        setName = "dataset.dat"
    weightName = raw_input("Path to save weight data? Press enter to default to: weightdata.wts.h5.\n")
    if weightName == "":
        print "Received no input, initializing weight as weightdata.wts.h5."
        weightName = "weightdata.wts.h5"
    print "Attempting to train classifier...\n"
    paramString = "python CNNscript.py "+setName+" "+weightName
    returnValue = subprocess.call(paramString, shell=True)
    if returnValue == 0:
        print("CNNscript.py was run successfully.")
    else:
        print("CNNscript.py encountered an error. Return value: {}. The program was called with {}.".format(returnValue, paramString))

def transmitData():
    scheme = raw_input("Which scheme would you like to modulate your transmission with? Avaliable schemes: [A]M, [F]M\n")
    if scheme[0].upper() != "A" and scheme[0].upper() != "F":
	print("Please enter either A or F to indicate which scheme to use.")
	transmitData()
    fileName = raw_input("Please enter the name of the WAV file you would like to transmit. If nothing is entered, the name 'song.wav' will be used.")
    if fileName == "":
        print "Received no input, initializing file name as 'song.wav'."
	fileName = "song.wav"
    print("Attempting to transmit data...")
    pyFile = None
    if scheme[0].upper() == "A":
    	pyFile = "AMtx.py"
    else:
	pyFile = "WBFMtx.py"
    paramString = "python"+" "+pyFile+" "+fileName
    returnValue = subprocess.call(paramString, shell=True)
    if returnValue == 0:
        print("{} was run successfully.".format(pyFile))
    else:
        print("{} encountered an error. Return value: {}".format(pyFile, returnValue))

def receiveData():
    decision = raw_input("Do you know the modulation scheme of the incoming signal? Type Y/N.")
    if decision.upper() == "Y":
	scheme = raw_input("Which scheme to demodulate with? Avaliable schemes: [A]M, [F]M\n")
	if scheme[0].upper() != "A" and scheme[0].upper() != "F":
            print("Please enter either A or F to indicate which scheme to use.")
            receiveData()
	fileName = raw_input("What would you like to name your received file? If no name is entered, the file will be named 'receivedFile.wav'")
	if fileName == "":
            print "Received no input, initializing file name as 'receivedFile.wav'."
            fileName = "receivedFile.wav"
        pyFile = None
        if scheme[0].upper() == "A":
            pyFile = "AMrx.py"
        else:
            pyFile = "WBFMrx.py"
        paramString = "python"+" "+pyFile+" "+fileName
        returnValue = subprocess.call(paramString, shell=True)
        if returnValue == 0:
            print("{} was run successfully.".format(pyFile))
        else:
            print("{} encountered an error. Return value: {}".format(pyFile, returnValue))
    elif decision.upper() == "N":
   	weights = raw_input("What weight data file would you like to use? If nothing is entered, 'weightdata.wts.h5' will be used.")
	if weights == "":
	    weights = "weightdata.wts.h5"
	paramString = "python CNNscript.py predict "+weights
	print("Attempting to identify scheme of incoming signal...")
        returnValue = subprocess.call(paramString, shell=True)
        if returnValue == 0:
            chooseFunc()
	else:
            print("CNNscript.py encountered an error. Return value: {}".format(returnValue))

    else:
	print("Please enter either Y or N to answer the question.")
	receiveData()

def main():
    chooseFunc()

if __name__ == '__main__':
    main()
