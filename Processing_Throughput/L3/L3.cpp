#include <uhd/device3.hpp>
#include <uhd/exception.hpp>
#include <uhd/rfnoc/block_ctrl.hpp>
#include <uhd/rfnoc/radio_ctrl.hpp>
#include <uhd/rfnoc/null_block_ctrl.hpp>
#include <uhd/rfnoc/ddc_block_ctrl.hpp>
#include <uhd/rfnoc/source_block_ctrl_base.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/types/sensors.hpp>
#include <uhd/utils/math.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/thread.hpp>

#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <chrono>
#include <complex>
#include <csignal>
#include <fstream>
#include <iostream>
#include <thread>
#include <locale>
#include <algorithm>
#include <string>
#include <fftw3.h>//must be after #include <complex>
#include <liquid/liquid.h>
#include "immintrin.h"
#include <cassert>

#include <x86intrin.h>    //AVX/SSE Extensions
#include <bits/stdc++.h>  //All main STD libraries
#define __AVX__ 1
#define __AVX2__ 1
#define __SSE__ 1
#define __SSE2__ 1
#define __SSE2_MATH__ 1
#define __SSE3__ 1
#define __SSE4_1__ 1
#define __SSE4_2__ 1
#define __SSE_MATH__ 1
#define __SSSE3__ 1

#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Optimization flags
#pragma GCC option("arch=native","tune=native","no-zero-upper") //Enable AVX
#pragma GCC target("avx")  //Enable AVX


namespace po = boost::program_options;
using uhd::rfnoc::radio_ctrl;


//ID of uhd blocs (from uhd_usrp_probe, the numbers have to be checked if more than one implementation of the same block is present on the fpga ex: Radio) 
const std::string ID_DDC = "DDC";
const std::string ID_RADIO0 = "0/Radio_0"; //Radio frontend select


//Add space in cout 
template<typename CharT>
struct Sep : public std::numpunct<CharT>
{
        virtual std::string do_grouping()      const   {return "\003";}
        virtual CharT       do_thousands_sep() const   {return ' ';}
};

const int64_t UPDATE_INTERVAL = 1; // 1 second update interval for BW summary

static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}


static inline void cabs_soa4(const float * re, const float *im, float * b) {
    __m128 r16 = _mm_set1_ps(16.0f);
    __m128 x4 = _mm_loadu_ps(re);
    __m128 y4 = _mm_loadu_ps(im);
    __m128 b4 = _mm_add_ps(_mm_mul_ps(x4,x4), _mm_mul_ps(y4,y4));
    __m128 r4 = _mm_div_ps(b4,r16);
    _mm_storeu_ps(b, b4);
}


/****************************************************************************
 * SIGINT handling
 ***************************************************************************/
static bool stop_signal_called = false;
void sig_int_handler(int)
{
    std::cout << "Exiting ..." << std::endl;
    stop_signal_called = true;
}

/****************************************************************************
 * recv_to_process
 ***************************************************************************/
template <typename samp_type>
const double recv_to_process(const std::string& file,
    const size_t samps_per_buff,
    const double rx_rate,
    double time_requested       = 0.0,
    int  processWait            = 0,
    bool bw_summary             = false,
    bool stats                  = false,
    bool enable_size_map        = false,
    bool continue_on_bad_packet = false)
{
    unsigned long long num_total_samps = 0;

    uhd::rx_metadata_t md;
    std::complex<float> buff[1024];
    std::complex<float> buff2[1024];
    float buff3[1024];
    float buff4[1024];
    __m256 v1,v2,v3;
    float tmp[4];

    std::complex<float>* out  = new std::complex<float>[samps_per_buff];

    std::ofstream outfile;
    if (not file.empty())  
        outfile.open(file.c_str(), std::ofstream::binary);

    bool overflow_message = true;

    fftwf_plan plan = fftwf_plan_dft_1d(int(samps_per_buff),
        reinterpret_cast<fftwf_complex*>(buff),
        reinterpret_cast<fftwf_complex*>(buff2),
        FFTW_FORWARD, 
        FFTW_PATIENT); //or  FFTW_PATIENT //FFTW_MEASURE

    const auto start_time = std::chrono::steady_clock::now();
    const auto stop_time = start_time + std::chrono::milliseconds(int64_t(1000 * time_requested));
    // Track time and samps between updating the BW summary
    auto last_update                     = start_time;
    unsigned long long last_update_samps = 0;

   const auto now = std::chrono::steady_clock::now();   
    while (not stop_signal_called and (std::chrono::steady_clock::now() <= stop_time)) 
    {
           
        num_total_samps+=1024;
    
    /////////////////////////////////////////////////////////////////////////
    /////////// Processing ////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////

       fftwf_execute(plan); 

       for (unsigned i = 0; i < 1024; i+= 4) 
        {
            float re[4] = {buff2[i].real(), buff2[i + 1].real(), buff2[i + 2].real(), buff2[i + 3].real()};
            float im[4] = {buff2[i].imag(), buff2[i + 1].imag(), buff2[i + 2].imag(), buff2[i + 3].imag()};
            cabs_soa4(re, im, buff3);
        }
               
       for (unsigned i = 15; i < 1024; i++)
        {
            v1=_mm256_loadu_ps(&(buff3[i-7]));
            v2=_mm256_loadu_ps(&(buff3[i-15]));
            v3=_mm256_add_ps(v1,v2);
            buff4[i]= _mm256_reduce_add_ps(v3);
        }

    }
    const auto actual_stop_time = std::chrono::steady_clock::now();
    fftwf_destroy_plan(plan); 

    const double actual_duration_seconds =
        std::chrono::duration<float>(actual_stop_time - start_time).count();

    const double rate = (double)num_total_samps / actual_duration_seconds;        
        return rate;
}




/****************************************************************************
 * main
 ***************************************************************************/
  int UHD_SAFE_MAIN(int argc, char* argv[])
{
    // variables to be set by po
    std::string args, file, format, ant, subdev, ref, wirefmt, streamargs, radio_args, block_id, block_args;
    size_t spp, radio_chan;
    double rate, freq, gain, bw, total_time, setup_time;
    std::cout.imbue(std::locale(std::cout.getloc(), new Sep <char>()));


    /////////////////////////////////////////////////////////////////////////
    /////////// Parameters ///////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////
    spp = 1024; //sample per packet    
    total_time = 10; //total number of seconds to receive, if 0 continuous mode (write on disak for each buff), if time specifie, IQ are stored in RAM and then wrote on disk
    format = "fc32";
    bool bw_summary = false; // periodically display short-term bandwidth
    bool stats = true;//        ("stats", "show average bandwidth on exit")
    bool enable_size_map = false;  //     ("sizemap", "track packet size and display breakdown on exit")
 
    bool continue_on_bad_packet = false;//       ("continue", "don't abort on a bad packet")
   
        
    /////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////
    // setup the program options
    po::options_description desc("Allowed options");
    // clang-format off
    desc.add_options()
        ("help", "help message")
        ("streamargs", po::value<std::string>(&streamargs)->default_value(""), "stream args")
        ("args", po::value<std::string>(&args)->default_value(""), "USRP device address args")
        ("null", "run without writing to file") 
    ;
    // clang-format on
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    // print the help message
    if (vm.count("help")) {std::cout << boost::format("Save samples to file %s") % desc << std::endl;return ~0;}

 
    /////////////////////////////////////////////////////////////////////////
    /////////// Set up benchmark /////////////////////////////////
    /////////////////////////////////////////////////////////////////////////
    
    
    
   double rcv_rate;

    std::ofstream myfile;
    myfile.open("../cppL3.txt");

    std::array<double,1> desiredRate = {200e6};
    int     processWait = 1;
    for (auto it = desiredRate.begin(); it != desiredRate.end(); ++it)
    {
        rate = *it;  
        // set the IF filter bandwidth
        bw = rate;
  
      

        // wait stabilize
        std::this_thread::sleep_for(std::chrono::milliseconds(int64_t(1000 * setup_time)));

        #define recv_to_process_args() \
            (   file,               \
                spp,                \
                rate,               \
                total_time,         \
                processWait,        \
                bw_summary,         \
                stats,              \
                enable_size_map,    \
                continue_on_bad_packet)
        float montecarlo_sum = 0;
        int montecarlo_num = 20;


        for (int montecarlo = 0 ; montecarlo < montecarlo_num ; ++montecarlo)
        {
            rcv_rate = recv_to_process<std::complex<float>> recv_to_process_args();  //benchmark lunched here
            montecarlo_sum += rcv_rate/montecarlo_num;
            std::cout << (rcv_rate/1e6) << " MSps" << std::endl;
        }        	         
        myfile << rate << "\t" << montecarlo_sum <<std::endl;
        std::cout << (montecarlo_sum/1e6) << " MSps" << std::endl;

    }
    myfile.close();
    /////////////////////////////////////////////////////////////////////////
    /////////// Finished ////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////    
    std::cout << std::endl << "Done!" << std::endl << std::endl;

    return EXIT_SUCCESS;
}
