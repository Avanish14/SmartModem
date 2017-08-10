# SmartModem

Transmitter/Receiver that employs analog/digital modulation and modulation scheme recognition with use of a USRP2.

More details about this project can be found at the Orbit Lab website: 

[www.orbit-lab.org/wiki/Other/Summer/2017/SpectrumClassification](www.orbit-lab.org/wiki/Other/Summer/2017/SpectrumClassification)

# Running the application

To work with the SmartModem, first download the project.
The following packages are needed:
* GNURadio
* TensorFlow
* Keras
* Numpy
* Sci-kit learn
* Matplotlib
* h5py

After installing the needed packages, enter the following command in the command-line: 
```javascript
python smartModem.py
```
Then, follow the command-line instructions.

# References 

* [RadioML](https://github.com/radioML/dataset) - code that we modified to build our neural network and to compile data
* [GNURadio](https://github.com/gnuradio/gnuradio) - SDR toolkit
* [TensorFlow](https://github.com/tensorflow/tensorflow) - Neural Network library

# License

This project is licensed under the MIT License.
