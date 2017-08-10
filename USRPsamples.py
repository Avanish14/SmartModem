#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Usrpsamples
# Generated: Wed Aug  9 12:31:34 2017
##################################################

from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import filter
from gnuradio import gr
from gnuradio import uhd
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import time, sys


class USRPsamples(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Usrpsamples")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 2500000

        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_source_0 = uhd.usrp_source(
        	",".join(("", "")),
        	uhd.stream_args(
        		cpu_format="fc32",
        		channels=range(1),
        	),
        )
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        self.uhd_usrp_source_0.set_center_freq(435000000, 0)
        self.uhd_usrp_source_0.set_gain(50, 0)
        self.uhd_usrp_source_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_source_0.set_bandwidth(50000, 0)
        self.rational_resampler_xxx_0 = filter.rational_resampler_ccc(
                interpolation=100000,
                decimation=2500000,
                taps=None,
                fractional_bw=None,
        )
        self.blocks_vector_sink_x_0 = blocks.vector_sink_c(1)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.rational_resampler_xxx_0, 0), (self.blocks_vector_sink_x_0, 0))    
        self.connect((self.uhd_usrp_source_0, 0), (self.rational_resampler_xxx_0, 0))    

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)


def main(top_block_cls=USRPsamples, options=None):

    tb = top_block_cls()
    tb.start()
    time.sleep(.10)
    data = tb.blocks_vector_sink_x_0.data()
    data = data[100:228] # Vector sample to identify modulation scheme with
    
    # Normalizing data
    #energy = np.sum((np.abs(data)))
    #data = data / energy
    redata = np.real(data)
    imdata = np.imag(data)
    data = []
    data.append(redata)
    data.append(imdata)
    print('!!!!!!!!!!!!!!!!!!!!!!!!')
    print(data)
    data = np.vstack(data)    
    print(data)
    tb.stop()
    tb.wait()
    #print(len(data))
    return data

if __name__ == '__main__':
    main()
