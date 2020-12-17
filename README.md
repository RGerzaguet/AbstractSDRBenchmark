

# Hardware configuration

i7-8850H CPU @2.60GHz with Intel SSE4.2, Intel AVX2
32GiB System Memory (4* 8GiB SODIMM DDR4 2667 MHz)
Ettus X310 with UHD 3.15 driver
FFT realized with FFTW 3.3.8 (full compile flag), plan generated in patient mode for all language

# Language configuration

**C++**
    Flag -gnu++11 -O3 -march=native -mavx 
    Gcc 7.5.0


**Julia**
    Flag -O3	
    Julia 1.5.0-rc1.0
    LoopVectorization (0.8.8)
    StaticArrays (0.12.3)

**Python **
    Flag -OO
    Python 3.6.9
    numba(0.50.1) 
    llvmlite(0.33.0)
    
    
# To compile and launch benchmark

**C++ **
build  && cmake .. && make && ./benchCpp

**Julia **
julia -O3 ****.jl

**Python **
python3 -OO benchPy.py
