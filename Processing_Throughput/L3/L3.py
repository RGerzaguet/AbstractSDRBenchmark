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
from numba import jit, vectorize, guvectorize
import ctypes
from ctypes import *
import llvmlite.binding as llvm
llvm.set_option('', 'opt -O0 -simplifycfg  -loop-rotate  -debug-only=loop-vectorize list_1.ll -S -o list_2.ll && opt -O3 -debug-only=loop-vectorize list_2.ll ')

CLOCK_TIMEOUT = 1000  # 1000mS timeout for external clock locking
INIT_DELAY = 0.05  # 50mS initial delay before transmit

def parse_args():
    """Parse the command line arguments"""
    description = """UHD Benchmark Rate (Python API)
        """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=description)
    parser.add_argument("-a", "--args", default="", type=str, help="single uhd device address args")
    parser.add_argument("-d", "--duration", default=4.0, type=float,
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
  
@jit(nopython=True,error_model="numpy") 
def num_abs3(x,y):
    for i in range(x.shape[0]):
        y[i] = x[i].real**2 + x[i].imag**2  
    

      
@jit(nopython=True,error_model="numpy",cache=True)
def windowMean2(x):    
    dt = x.dtype
    y = numpy.empty_like(x)
    for i in range(16,1024):      
        y[i] = (x[i]+x[i-1]+x[i-2]+x[i-3]+x[i-4]+x[i-5]+x[i-6]+x[i-7]+x[i-8]+x[i-9]+x[i-10]+x[i-11]+x[i-12]+x[i-13]+x[i-14]+x[i-15])/dt.type(16)     
    return y
        
        
@jit(nopython=True,error_model="numpy") 
def windowMean3(x,y):    
    dt = x.dtype
    for i in range(16,1024):      
        y[i] = (x[i]+x[i-1]+x[i-2]+x[i-3]+x[i-4]+x[i-5]+x[i-6]+x[i-7]+x[i-8]+x[i-9]+x[i-10]+x[i-11]+x[i-12]+x[i-13]+x[i-14]+x[i-15])/dt.type(16)     
        

def benchmark_rx_rate(timer_elapsed_event, rx_statistics,flag,recv_buffer,fft_buffer,mean_buffer,abs_buffer,fft_plan):
    """Benchmark the receive chain"""
    
   # Setup the statistic counters
    num_rx_samps = 0
   
    # Receive until we get the signal to stop
    while not timer_elapsed_event.is_set():
        try:
            num_rx_samps += 1024
            fft_plan.execute()
            fft_buffer = fft_plan.get_output_array()            
            num_abs3(fft_buffer,abs_buffer) 
            windowMean3(abs_buffer,mean_buffer)

        except RuntimeError as ex:
            return
          
    # Return the statistics to the main thread
    rx_statistics["num_rx_samps"] = num_rx_samps


def main():
    """Run the benchmarking tool"""
    args = parse_args()
    threads = []
    # Make a signal for the threads to stop running
    # Create a dictionary for the RX statistics
    # Note: we're going to use this without locks, so don't access it from the main thread until
    #       the worker has joined
    rx_statistics = {}
    # Spawn the transmit test thread
    desiredRate = [200e6];
    f = open("pythonL3.txt", "w")

    flag = 1;
    max_samps_per_packet = 1024
    recv_buffer = pyfftw.n_byte_align_empty(max_samps_per_packet, dtype='complex64', n=16)
     
    fft_buffer = numpy.zeros(max_samps_per_packet,dtype=numpy.complex64, order='F')  
    mean_buffer = numpy.zeros(max_samps_per_packet,dtype=numpy.float32, order='F')
    abs_buffer =  numpy.zeros(max_samps_per_packet,dtype=numpy.float32, order='F')
    
    fft_plan = pyfftw.FFTW(recv_buffer, fft_buffer, flags=('FFTW_PATIENT',), threads=1) 

    
    montecarlo_num = 20;
    for r in desiredRate:       

        # Spawn the receive test thread
        e = r
        print(f'Testing receive rate  {e/1e6} Msps');
        montecarlo_sum = 0;
        for mt in range(1,montecarlo_num+1):
            
            quit_event = threading.Event()
            rx_thread = threading.Thread(target=benchmark_rx_rate,
                                        args=(quit_event,
                                            rx_statistics,flag,recv_buffer,fft_buffer,mean_buffer,abs_buffer,fft_plan))
            threads.append(rx_thread)
            

            
            rx_thread.start()
            rx_thread.setName("bmark_rx_stream")
        
            time.sleep(args.duration)
            # Interrupt and join the threads
            quit_event.set()
            for thr in threads:
                thr.join()

            numsamps = rx_statistics.get("num_rx_samps", 0);    
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
