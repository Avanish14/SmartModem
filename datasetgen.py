#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Ngmodtesting
# Generated: Wed Jun 28 16:58:23 2017
##################################################

from gnuradio import analog
from gnuradio import blocks
from gnuradio import channels
from gnuradio import digital
from gnuradio import eng_notation
from gnuradio import filter
from gnuradio import gr
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import numpy as np, numpy
from time import gmtime, strftime
import time
import sys
import cPickle
import sip
import random


class ngModTesting(gr.top_block):
    def __init__(self, modScheme, snr):
        self.rmsValue = rmsValue = 1.0  # RMS value to calculate noise voltage with
        print "Generating", modScheme," with ",snr,"SNR"
	floatSNR = float(snr)
        self.noiseLevel = noiseLevel = rmsValue / (
        10 ** (floatSNR / 20.0))  # Sets noise voltage of channel model based on the SNR parameter
        gr.top_block.__init__(self, "Ngmodtesting")

        ##################################################
        # Variables
        ##################################################

        self.randNumLimit = randNumLimit = 256  # 256 to generate 0-255, put as 1 if only want noise
        self.samp_rate = samp_rate = 300000  # Set sampling rate for throttle blocks, was 90000 before
        self.modBPSK = modBPSK = digital.constellation_rect(([-1, 1]), ([0, 1]), 4, 2, 2, 1, 1).base()
        self.modQPSK = modQPSK = digital.constellation_rect(([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j]), ([0, 1, 2, 3]), 4, 2,
                                                            2, 1, 1).base()
        # 00: -1-1j
        # 01: -1+1j
        # 10: +1-1j
        # 11: +1+1j
        self.mod16QAM = mod16QAM = digital.constellation_rect(([-1 - 1j, -1 - 2j, -2 - 1j, -2 - 2j, -1 + 1j, -1 + 2j,
                                                                -2 + 1j, -2 + 2j, 1 - 1j, 1 - 2j, 2 - 1j, 2 - 2j,
                                                                1 + 1j, 1 + 2j, 2 + 1j, 2 + 2j]),
                                                              ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                                                              4, 2, 2, 1, 1).base()
        symbols32QAM = range(0, 32, 1)
        self.mod32QAM = mod32QAM = digital.constellation_rect(([-1 - 1j, 1 - 1j, -1 + 1j, 1 + 1j, -1 - 2j, 1 - 2j,
                                                                -1 + 2j, 1 + 2j, -1 - 3j, 1 - 3j, -1 + 3j, 1 + 3j,
                                                                -2 - 1j, 2 - 1j, -2 + 1j, 2 + 1j, -2 - 2j, 2 - 2j,
                                                                -2 + 2j, 2 + 2j, -2 - 3j, 2 - 3j, -2 + 3j, 2 + 3j,
                                                                -3 - 1j, 3 - 1j, -3 + 1j, 3 + 1j, -3 - 2j, 3 - 2j,
                                                                -3 + 2j, 3 + 2j]), symbols32QAM, 4, 2, 2, 1, 1).base()
        symbols64QAM = range(0, 64, 1)
        self.mod64QAM = mod64QAM = digital.constellation_rect(([-1 - 1j, 1 - 1j, -1 + 1j, 1 + 1j, -1 - 2j, 1 - 2j,
                                                                -1 + 2j, 1 + 2j, -1 - 3j, 1 - 3j, -1 + 3j, 1 + 3j,
                                                                -1 - 4j, 1 - 4j, -1 + 4j, 1 + 4j, -2 - 1j, 2 - 1j,
                                                                -2 + 1j, 2 + 1j, -2 - 2j, 2 - 2j, -2 + 2j, 2 + 2j,
                                                                -2 - 3j, 2 - 3j, -2 + 3j, 2 + 3j, -2 - 4j, 2 - 4j,
                                                                -2 + 4j, 2 + 4j, -3 - 1j, 3 - 1j, -3 + 1j, 3 + 1j,
                                                                -3 - 2j, 3 - 2j, -3 + 2j, 3 + 2j, -3 - 3j, 3 - 3j,
                                                                -3 + 3j, 3 + 3j, -3 - 4j, 3 - 4j, -3 + 4j, 3 + 4j,
                                                                -4 - 1j, 4 - 1j, -4 + 1j, 4 + 1j, -4 - 2j, 4 - 2j,
                                                                -4 + 2j, 4 + 2j, -4 - 3j, 4 - 3j, -4 + 3j, 4 + 3j,
                                                                -4 - 4j, 4 - 4j, -4 + 4j, 4 + 4j]), symbols64QAM, 4, 2,
                                                              2, 1, 1).base()
        symbols128QAM = range(0, 128, 1)
        list = [1, 2, 3, 4, 5, 6]
        points128QAM = []
        counter = 0
        for num in list:
            for num2 in list:
                if num >= 5 and num2 >= 5:
                    continue
                else:
                    y = 1 + 1j
                    z = -1 * y.real * num - y.imag * num2 * 1j
                    points128QAM.append(z)
                    z = -1 * y.real * num + y.imag * num2 * 1j
                    points128QAM.append(z)
                    z = y.real * num - y.imag * num2 * 1j
                    points128QAM.append(z)
                    z = y.real * num + y.imag * num2 * 1j
                    points128QAM.append(z)
        self.mod128QAM = mod128QAM = digital.constellation_rect(points128QAM, symbols128QAM, 4, 2, 2, 1, 1).base()
        symbols256QAM = range(0, 256, 1)
        list = range(1, 9, 1)  # 0..8
        points256QAM = []
        counter = 0
        for num in list:
            for num2 in list:
                y = 1 + 1j
                z = -1 * y.real * num - y.imag * num2 * 1j
                points256QAM.insert(counter, z)
                counter += 1
                z = -1 * y.real * num + y.imag * num2 * 1j
                points256QAM.insert(counter, z)
                counter += 1
                z = y.real * num - y.imag * num2 * 1j
                points256QAM.insert(counter, z)
                counter += 1
                z = y.real * num + y.imag * num2 * 1j
                points256QAM.insert(counter, z)
                counter += 1
        self.mod256QAM = mod256QAM = digital.constellation_rect(points256QAM, symbols256QAM, 4, 2, 2, 1, 1).base()
        self.modNoise = modNoise = digital.constellation_rect(([0, -1 + 1j, 1 - 1j, 1 + 1j]), ([0, 1, 2, 3]), 4, 2, 2,
                                                              1, 1).base()

        self.sink = None
        self.constObj = constObj = None
        if modScheme == "FM":
            self.sink = self.fmGen(noiseLevel)
            return
        elif modScheme == "AM-SSB":
            self.sink = self.amGen("SSB", noiseLevel)
            return
        elif modScheme == "AM-DSBSC":
            self.sink = self.amGen("DSBSC", noiseLevel)
            return
        elif modScheme == "AM-DSBFC":
            self.sink = self.amGen("DSBFC", noiseLevel)
            return
        elif modScheme == "16QAM":
            self.constObj = constObj = mod16QAM
        elif modScheme == "32QAM":
            self.constObj = constObj = mod32QAM
        elif modScheme == "64QAM":
            self.constObj = constObj = mod64QAM
        elif modScheme == "128QAM":
            self.constObj = constObj = mod128QAM
        elif modScheme == "256QAM":
            self.constObj = constObj = mod256QAM
        elif modScheme == "QPSK":
            self.constObj = constObj = modQPSK
        elif modScheme == "BPSK":
            self.constObj = constObj = modBPSK
        elif modScheme.upper() == "NOISE":  
            self.constObj = constObj = modNoise
            randNumLimit = 1
        else:
            print("Error: Hardcoded schemes does not match accepted schemes. Offending input: {}'".format(modScheme))
            sys.exit(-1)

        self.channel = channels.channel_model(
            noise_voltage=noiseLevel,  # AWGN noise level as a voltage (edit depending on SNR)
            frequency_offset=0.0,  # No frequency offset
            epsilon=1.0,  # 1.0 to keep no difference between sampling rates of clocks of transmitter and receiver
            noise_seed=0,  # Normal random number generator for noise
            block_tags=False  # Only needed for multipath
        )

        ##################################################
        # Blocks
        ##################################################
        self.digital_constellation_modulator_0 = digital.generic_mod(
            constellation=constObj,
            differential=True,  # True for PSK/QAM, allows modulation based on successive symbols
            samples_per_symbol=2,  # For every symbol, adds (samples_per_symbol)-1 zeroes after the original symbol
            pre_diff_code=False,
            excess_bw=0.35,
            verbose=False,
            log=False,
        )
        self.sink = blocks.vector_sink_c(1)
        self.blocks_vector_sink_x_5 = blocks.vector_sink_f(1)
        self.blocks_vector_sink_x_4 = blocks.vector_sink_c(1)
        self.blocks_vector_sink_x_1 = blocks.vector_sink_b(1)
        self.blocks_vector_sink_x_0 = blocks.vector_sink_c(1)
        self.blocks_throttle_1 = blocks.throttle(gr.sizeof_char * 1, samp_rate)#, True)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex * 1, samp_rate)#, True)
        self.blocks_stream_to_vector_0 = blocks.stream_to_vector(gr.sizeof_char * 1, 1)
        self.numSamples = numSamples = 10000000  # Number of samples for random number generator
        self.isRepeat = isRepeat = True  # Repeat random number generator?
        self.analog_random_source_x_0 = blocks.vector_source_b(
            map(int, numpy.random.randint(0, randNumLimit, numSamples)), isRepeat)

        self.blocks_rms_xx_0 = blocks.rms_cf(0.0001)



        ##################################################
        # Connections
        ##################################################
        self.sink2 = self.blocks_vector_sink_x_1
        self.sink5 = self.blocks_vector_sink_x_4
        self.rmssink = self.blocks_vector_sink_x_5

        # Byte Stream:
        self.connect((self.analog_random_source_x_0, 0), (self.blocks_stream_to_vector_0, 0))
        self.connect((self.blocks_stream_to_vector_0, 0), (self.blocks_throttle_1, 0))
        self.connect((self.blocks_throttle_1, 0), (self.sink2, 0))

        # Constellation Block Raw Output:
        self.connect((self.analog_random_source_x_0, 0), (self.digital_constellation_modulator_0, 0))
        self.connect((self.digital_constellation_modulator_0, 0), (self.sink5, 0))

        # Modulated Signal + Noise Path:
        self.connect((self.digital_constellation_modulator_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.channel, 0))
        self.connect((self.channel, 0), (self.sink, 0))

        # Get RMS
        self.connect((self.digital_constellation_modulator_0, 0), (self.blocks_rms_xx_0, 0))
        self.connect((self.blocks_rms_xx_0, 0), (self.rmssink, 0))

        def closeEvent(self, event):
            self.settings = Qt.QSettings("GNU Radio", "ngModTesting")
            self.settings.setValue("geometry", self.saveGeometry())
            event.accept()

    def fmGen(self, noiseLevel):

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 300000

        ##################################################
        # Blocks
        ##################################################


        self.blocks_wavfile_source_0 = blocks.wavfile_source(sys.argv[3],
                                                             True)
        self.sink = blocks.vector_sink_c(1)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex * 1, samp_rate)#, True)
        self.blocks_multiply_const_vxx_1 = blocks.multiply_const_vcc((1,))
        self.analog_wfm_tx_0 = analog.wfm_tx(
            audio_rate=44100,
            quad_rate=176400,
            tau=75e-6,
            max_dev=5e3,
     #       fh=-1.0,
        )

        self.channel = channels.channel_model(
            noise_voltage=noiseLevel,  # AWGN noise level as a voltage (edit depending on SNR)
            frequency_offset=0.0,  # No frequency offset
            epsilon=1.0,  # 1.0 to keep no difference between sampling rates of clocks of transmitter and receiver
            noise_seed=0,  # Normal random number generator for noise
            block_tags=False  # Only needed for multipath
        )
        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_wavfile_source_0, 0), (self.analog_wfm_tx_0, 0))
        self.connect((self.analog_wfm_tx_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.blocks_multiply_const_vxx_1, 0))
        self.connect((self.blocks_multiply_const_vxx_1, 0), (self.channel, 0))
        self.connect((self.channel, 0), (self.sink, 0))

        return self.sink

    def amGen(self, type, noiseLevel):

        ##################################################
        # Variables
        ##################################################
        self.add = add = 0  # set to 1 if not suppressing carrier
        if type == "DSBFC":
            add = 1

        self.samp_rate = samp_rate = 300000

        ##################################################
        # Blocks
        ##################################################
        self.sink = blocks.vector_sink_c(1)
        self.blocks_wavfile_source_0 = blocks.wavfile_source(sys.argv[3],
                                                             True)
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)
        self.blocks_multiply_xx_0 = blocks.multiply_vcc(1)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_vcc((1,))
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)
        self.blocks_add_const_vxx_0 = blocks.add_const_vcc((add,))
        self.analog_sig_source_x_1 = analog.sig_source_c(samp_rate, analog.GR_COS_WAVE, 100000, 1, 0)
        self.blocks_throttle_0 = blocks.throttle(gr.sizeof_gr_complex * 1, samp_rate)#, True)
        self.channel = channels.channel_model(
            noise_voltage=noiseLevel,  # AWGN noise level as a voltage (edit depending on SNR)
            frequency_offset=0.0,  # No frequency offset
            epsilon=1.0,  # 1.0 to keep no difference between sampling rates of clocks of transmitter and receiver
            noise_seed=0,  # Normal random number generator for noise
            block_tags=False  # Only needed for multipath
        )
        self.band_pass_filter_0 = filter.fir_filter_ccf(1, firdes.band_pass(
            1, samp_rate, 100000, 150000, 10000, firdes.WIN_KAISER, 6.76))  # used only for SSB
        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_wavfile_source_0, 0), (self.blocks_float_to_complex_0, 0))
        self.connect((self.blocks_float_to_complex_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.analog_sig_source_x_1, 0), (self.blocks_multiply_xx_0, 1))
        self.connect((self.blocks_add_const_vxx_0, 0), (self.blocks_multiply_xx_0, 0))
        if type == "SSB":
            self.connect((self.blocks_multiply_xx_0, 0), (self.band_pass_filter_0, 0))
            self.connect((self.band_pass_filter_0, 0), (self.blocks_throttle_0, 0))
        else:
            self.connect((self.blocks_multiply_xx_0, 0), (self.blocks_throttle_0, 0))
        self.connect((self.blocks_throttle_0, 0), (self.channel, 0))
        self.connect((self.channel, 0), (self.sink, 0))
        return self.sink


def main():
    print(sys.argv)
    dataset = {}
    nvecs_per_key = 1000
    vec_length = 128    
    if sys.argv[1].upper() == "ALL":
        modList = ["FM", "AM-SSB", "AM-DSBFC", "AM-DSBSC", "16QAM", "128QAM", "Noise"]
    else:
        modList = sys.argv[1].split(',')
    if sys.argv[2].upper() == "ALL":    
	snrList = [-20,-10,0,5,10,15,20]
    else:
        snrList = sys.argv[2].split(',')
    	for i in range(len(snrList)):
		snrList[i] = int(snrList[i])
    # Running gnuradio script
    for mod in modList:
        for snr in snrList:
            dataset[(mod, snr)] = np.zeros([nvecs_per_key, 2, vec_length], dtype=np.float32)
            tb = ngModTesting(mod, snr)
            tb.start()

            def quitting():
                tb.stop()
                tb.wait()

            time.sleep(1)

            dataToRecord = tb.sink.data()  
            #np.set_printoptions(threshold=numpy.nan) # uncomment if printing vectors
            # Serializing/Saving Data
            modvec_indx = 0
            raw_output_vector = np.array(dataToRecord, dtype=np.complex64)
            # start the sampler some random time after channel model transients (arbitrary values here)
            sampler_indx = random.randint(50, 500)
            while modvec_indx < nvecs_per_key:
                sampled_vector = raw_output_vector[sampler_indx:sampler_indx + vec_length]
                # Normalize the energy in this vector to be 1
                energy = np.sum((np.abs(sampled_vector)))
                sampled_vector = sampled_vector / energy
                #if (modvec_indx == 0): # prints one of the pickled vectors
                #    print "sample vector:", sampled_vector
                #    print "-------------------------------------"
                if len(sampled_vector)==0:
                    sampler_indx = random.randint(50, 500)
                    continue
                dataset[(mod, snr)][modvec_indx, 0, :] = np.real(sampled_vector)
                dataset[(mod, snr)][modvec_indx, 1, :] = np.imag(sampled_vector)
                # bound the upper end very high so it's likely we get multiple passes through
                # independent channels
                sampler_indx = random.randint(vec_length, len(raw_output_vector)-vec_length-1)
                modvec_indx += 1
            if modvec_indx == nvecs_per_key:
                insufficient_modsnr_vectors = False
            else:
                print("found only {} out of {} vectors: need to increase numSamples in byte stream to get more".format(
                    modvec_indx, nvecs_per_key))

    #currentTime = strftime("%Y-%m-%d %Hh%Mm%Ss", gmtime())
    #print "UTC time: ", currentTime
    fileName = sys.argv[4]

    try:
        print "Attempting to save to file:", fileName
        cPickle.dump(dataset, file(fileName, "wb"))
    except EOFError:
        pass
    else:
        print "Completed without issue."


if __name__ == '__main__':
    main()
