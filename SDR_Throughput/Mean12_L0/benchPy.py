#!/usr/bin/env python3
"""
Benchmark rate using Python API
"""

import argparse
from datetime import datetime, timedelta
import sys
import time
import threading
import logging
import numpy
import uhd
import pdb
import math
import pyfftw
import numba
from numba import jit

CLOCK_TIMEOUT = 1000  # 1000mS timeout for external clock locking
INIT_DELAY = 0.05  # 50mS initial delay before transmit

def parse_args():
    """Parse the command line arguments"""
    description = """UHD Benchmark Rate (Python API)
        """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=description)
    parser.add_argument("-a", "--args", default="", type=str, help="single uhd device address args")
    parser.add_argument("-d", "--duration", default=10.0, type=float,
                        help="duration for the test in seconds")
    parser.add_argument("--rx_stream_args",
                        help="stream args for RX streamer", default="")
    return parser.parse_args()


class LogFormatter(logging.Formatter):
    """Log formatter which prints the timestamp with fractional seconds"""
    @staticmethod
    def pp_now():
        """Returns a formatted string containing the time of day"""
        now = datetime.now()
        return "{:%H:%M}:{:05.2f}".format(now, now.second + now.microsecond / 1e6)
        # return "{:%H:%M:%S}".format(now)

    def formatTime(self, record, datefmt=None):
        converter = self.converter(record.created)
        if datefmt:
            formatted_date = converter.strftime(datefmt)
        else:
            formatted_date = LogFormatter.pp_now()
        return formatted_date



def benchmark_rx_rate(usrp, rx_streamer, timer_elapsed_event, rx_statistics,flag,recv_buffer,fft_buffer,mean_buffer,abs_buffer,fft_plan):
    """Benchmark the receive chain"""

    # Make a receive buffer
    num_channels = rx_streamer.get_num_channels()


    metadata = uhd.types.RXMetadata()

    # Craft and send the Stream Command
    if flag == 1:
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
        stream_cmd.stream_now = (num_channels == 1)
        stream_cmd.time_spec = uhd.types.TimeSpec(usrp.get_time_now().get_real_secs() + INIT_DELAY)
        rx_streamer.issue_stream_cmd(stream_cmd)

    # Setup the statistic counters
    num_rx_samps = 0


    rate = usrp.get_rx_rate()
    # Receive until we get the signal to stop
    while not timer_elapsed_event.is_set():
        try:
            num_rx_samps += rx_streamer.recv(recv_buffer, metadata)
            fft_plan.execute()
            fft_buffer = fft_plan.get_output_array()
            abs_buffer = fft_buffer.real**2 + fft_buffer.imag**2
            for i in range(12,1024):
                mean_buffer[i] = (abs_buffer[i]+abs_buffer[i-1]+abs_buffer[i-2]+abs_buffer[i-3]+abs_buffer[i-4]+abs_buffer[i-5]+abs_buffer[i-6]+abs_buffer[i-7]+abs_buffer[i-8]+abs_buffer[i-9]+abs_buffer[i-10]+abs_buffer[i-11])/12.0

        except RuntimeError as ex:
            return

    # Return the statistics to the main thread
    rx_statistics["num_rx_samps"] = num_rx_samps


def print_statistics(rx_statistics, tx_statistics, tx_async_statistics):
    """Print TRX statistics in a formatted block"""
    logger.debug("RX Statistics Dictionary: %s", rx_statistics)
    logger.debug("TX Statistics Dictionary: %s", tx_statistics)
    logger.debug("TX Async Statistics Dictionary: %s", tx_async_statistics)
    # Print the statistics
    statistics_msg = """Benchmark rate summary:
    Num received samples:     {}
    Num dropped samples:      {}
    Num overruns detected:    {}
    Num transmitted samples:  {}
    Num sequence errors (Tx): {}
    Num sequence errors (Rx): {}
    Num underruns detected:   {}
    Num late commands:        {}
    Num timeouts (Tx):        {}
    Num timeouts (Rx):        {}""".format(
        rx_statistics.get("num_rx_samps", 0),
        rx_statistics.get("num_rx_dropped", 0),
        rx_statistics.get("num_rx_overruns", 0),
        tx_statistics.get("num_tx_samps", 0),
        tx_async_statistics.get("num_tx_seqerr", 0),
        rx_statistics.get("num_rx_seqerr", 0),
        tx_async_statistics.get("num_tx_underrun", 0),
        rx_statistics.get("num_rx_late", 0),
        tx_async_statistics.get("num_tx_timeouts", 0),
        rx_statistics.get("num_rx_timeouts", 0))
    logger.info(statistics_msg)


def main():
    """Run the benchmarking tool"""
    args = parse_args()
    # Setup a usrp device
    usrp = uhd.usrp.MultiUSRP(args.args)
    rx_channels = [0]
    usrp.set_time_now(uhd.types.TimeSpec(0.0))

    threads = []
    # Make a signal for the threads to stop running
    # Create a dictionary for the RX statistics
    # Note: we're going to use this without locks, so don't access it from the main thread until
    #       the worker has joined
    rx_statistics = {}
    # Spawn the transmit test thread
    desiredRate = [2e6,10e6,20e6,30e6,40e6,50e6,75e6,100e6,200e6];
    f = open("../data_benchmark_python_mean12_quick.txt", "w")
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = rx_channels
    st_args.args = uhd.types.DeviceAddr(args.rx_stream_args)
    rx_streamer = usrp.get_rx_stream(st_args)


    flag = 1;
    max_samps_per_packet = 1024
    recv_buffer = pyfftw.empty_aligned(max_samps_per_packet, dtype='complex64', n=16)
    fft_buffer  = pyfftw.empty_aligned(max_samps_per_packet, dtype='complex64', n=16)
    mean_buffer = numpy.zeros(max_samps_per_packet,dtype='float64')
    abs_buffer =  numpy.zeros(max_samps_per_packet,dtype='float64')
    fft_plan = pyfftw.FFTW(recv_buffer, fft_buffer, flags=('FFTW_PATIENT',), threads=1)


    montecarlo_num = 20;
    for r in desiredRate:


        # Spawn the receive test thread
        usrp.set_rx_rate(r)
        e = usrp.get_rx_rate();
        print(f'Testing receive rate  {e/1e6} Msps');
        montecarlo_sum = 0;
        for mt in range(1,montecarlo_num+1):

            quit_event = threading.Event()
            rx_thread = threading.Thread(target=benchmark_rx_rate,
                                        args=(usrp, rx_streamer, quit_event,
                                            rx_statistics,flag,recv_buffer,fft_buffer,mean_buffer,abs_buffer,fft_plan))
            threads.append(rx_thread)



            rx_thread.start()
            rx_thread.setName("bmark_rx_stream")

            time.sleep(args.duration)
            # Interrupt and join the threads
            quit_event.set()
            for thr in threads:
                thr.join()

            #print_statistics(rx_statistics, tx_statistics, tx_async_statistics)
            numsamps = rx_statistics.get("num_rx_samps", 0);
            rx_streamer.issue_stream_cmd(uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont))

            #print(f'Received samples = {numsamps}');
            rate = numsamps / args.duration;
            montecarlo_sum = montecarlo_sum + rate/montecarlo_num
            print(f'');
            print(f'Rate {rate/1e6} MHz');

        print(f'Rate {e/1e6} - Rate {montecarlo_sum/1e6} MHz');
        f.write(f'{e} \t {montecarlo_sum}\n');




    return True


if __name__ == "__main__":
    # Setup the logger with our custom timestamp formatting
    global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console = logging.StreamHandler()
    logger.addHandler(console)
    formatter = LogFormatter(fmt='[%(asctime)s] [%(levelname)s] (%(threadName)-10s) %(message)s')
    console.setFormatter(formatter)

    # Vamos, vamos, vamos!
    sys.exit(not main())
