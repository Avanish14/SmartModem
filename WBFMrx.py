#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Wbfmrx
# Generated: Tue Aug  8 18:29:53 2017
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


class WBFMrx(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Wbfmrx")

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
        self.uhd_usrp_source_0.set_gain(40, 0)
        self.uhd_usrp_source_0.set_antenna('TX/RX', 0)
        self.uhd_usrp_source_0.set_bandwidth(100000, 0)
        self.rational_resampler_xxx_1 = filter.rational_resampler_ccc(
                interpolation=192000,
                decimation=250000,
                taps=None,
                fractional_bw=None,
        )
        self.low_pass_filter_0 = filter.fir_filter_ccf(10, firdes.low_pass(
        	1, samp_rate, 42000, 2100, firdes.WIN_HAMMING, 6.76))
        self.blocks_wavfile_sink_0 = blocks.wavfile_sink(sys.argv[1], 1, 48000, 8)
        self.analog_wfm_rcv_0 = analog.wfm_rcv(
        	quad_rate=192000,
        	audio_decimation=4,
        )

        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_wfm_rcv_0, 0), (self.blocks_wavfile_sink_0, 0))    
        self.connect((self.low_pass_filter_0, 0), (self.rational_resampler_xxx_1, 0))    
        self.connect((self.rational_resampler_xxx_1, 0), (self.analog_wfm_rcv_0, 0))    
        self.connect((self.uhd_usrp_source_0, 0), (self.low_pass_filter_0, 0))    

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)
        self.low_pass_filter_0.set_taps(firdes.low_pass(1, self.samp_rate, 42000, 2100, firdes.WIN_HAMMING, 6.76))


def main(top_block_cls=WBFMrx, options=None):

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
