This file proposes some benchmark performance in both C++, Python and Julia.
The purpose is to estimate the output throughput both when a Software Defined Radio (SDR) and in direct timing.

# Hardware configuration

The following configuration has been used
- i7-8850H CPU @2.60GHz with Intel SSE4.2, Intel AVX2
- 32GiB System Memory (4* 8GiB SODIMM DDR4 2667 MHz)
- Ettus X310 with UHD 3.15 driver
- FFT realized with FFTW 3.3.8 (full compile flag), plan generated in patient mode for all language

# Language configuration

We expose here the specific properties uses for each langage.

**C++**
- Flag -gnu++11 -O3 -march=native -mavx 
- Gcc 7.5.0


**Julia**
- Flag -O3	
- Julia 1.5.0-rc1.0
- LoopVectorization (0.8.8)
- StaticArrays (0.12.3)
- FFTW 1.2.4

**Python**
- Flag -OO
- Python 3.6.9
- numba (0.50.1) 
- llvmlite (0.33.0)

    
# To compile and launch benchmark

We expose here the steps to compile the codes and obtain the performance.
There is 2 main folders
- Receive: Different code to be used with a SDR (i.e X310) to monitor link between X310 rate and processing rate 
- SoftThroughput : Maximal rate in different langage with no interface with SDR with 3 optimisations level

**C++**

To compile the file and run the bench use the follwing command:
```build  && cmake .. && make && ./benchCpp```

**Julia**

In Julia, the julia benchmark files can be launched in a fresh REPL or alternatively by 
```julia -O3 benchmark_rate_severalRate.jl```

**Python**

The file to be launched in Python is
```python3 -OO benchPy.py```
