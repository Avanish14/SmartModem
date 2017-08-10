#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Amtx
# Generated: Tue Aug  8 19:02:26 2017
##################################################

from gnuradio import analog
from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import filter
from gnuradio import gr
from gnuradio import uhd
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import time
import sys

class AMtx(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Amtx")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 2500000

        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_sink_0 = uhd.usrp_sink(
        	",".join(("", "")),
        	uhd.stream_args(
        		cpu_format="fc32",
        		channels=range(1),
        	),
        )
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        self.uhd_usrp_sink_0.set_center_freq(435000000, 0)
        self.uhd_usrp_sink_0.set_gain(80, 0)
        self.uhd_usrp_sink_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_sink_0.set_bandwidth(100000, 0)
        self.rational_resampler_xxx_0 = filter.rational_resampler_ccc(
                interpolation=2500000,
                decimation=44100,
                taps=None,
                fractional_bw=None,
        )
        self.blocks_wavfile_source_0 = blocks.wavfile_source(sys.argv[1], True)
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)
        self.analog_sig_source_x_0 = analog.sig_source_f(44100, analog.GR_COS_WAVE, 0, 1, 0)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_sig_source_x_0, 0), (self.blocks_float_to_complex_0, 1))    
        self.connect((self.blocks_float_to_complex_0, 0), (self.rational_resampler_xxx_0, 0))    
        self.connect((self.blocks_wavfile_source_0, 0), (self.blocks_float_to_complex_0, 0))    
        self.connect((self.rational_resampler_xxx_0, 0), (self.uhd_usrp_sink_0, 0))    

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)


def main(top_block_cls=AMtx, options=None):

    tb = top_block_cls()
    tb.start()
    print('Transmitting on ' + str(tb.uhd_usrp_sink_0.get_center_freq()) + 'Hz with a channel bandwidth of ' + str(tb.uhd_usrp_sink_0.get_bandwidth()) + 'Hz')
    try:
        raw_input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    main()
