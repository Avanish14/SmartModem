#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Amrx
# Generated: Tue Aug  8 20:51:18 2017
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
import time, sys


class AMrx(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Amrx")

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
        self.uhd_usrp_source_0.set_gain(80, 0)
        self.uhd_usrp_source_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_source_0.set_bandwidth(100000, 0)
        self.rational_resampler_xxx_0 = filter.rational_resampler_ccc(
                interpolation=44100,
                decimation=2500000,
                taps=None,
                fractional_bw=None,
        )
        self.blocks_wavfile_sink_0 = blocks.wavfile_sink(sys.argv[1], 1, 44100, 8)
        self.analog_am_demod_cf_0 = analog.am_demod_cf(
        	channel_rate=44100,
        	audio_decim=1,
        	audio_pass=20000,
        	audio_stop=21000,
        )
        self.analog_agc2_xx_0 = analog.agc2_cc(.1, 1e-6, 1.0, 0)
        self.analog_agc2_xx_0.set_max_gain(5)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_agc2_xx_0, 0), (self.rational_resampler_xxx_0, 0))    
        self.connect((self.analog_am_demod_cf_0, 0), (self.blocks_wavfile_sink_0, 0))    
        self.connect((self.rational_resampler_xxx_0, 0), (self.analog_am_demod_cf_0, 0))    
        self.connect((self.uhd_usrp_source_0, 0), (self.analog_agc2_xx_0, 0))    

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)


def main(top_block_cls=AMrx, options=None):

    tb = top_block_cls()
    tb.start()
    print('Receiving on ' + str(tb.uhd_usrp_source_0.get_center_freq()) + 'Hz with a channel bandwidth of ' + str(tb.uhd_usrp_source_0.get_bandwidth()) + 'Hz')
    try:
        raw_input('Press Enter to quit: ')
    except EOFError:
        pass
    tb.stop()
    tb.wait()
    print('.wav file generated')

if __name__ == '__main__':
    main()
