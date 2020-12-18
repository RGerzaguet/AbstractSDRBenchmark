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
    float tmp[4];

    std::complex<float>* out  = new std::complex<float>[samps_per_buff];

    std::ofstream outfile;
    if (not file.empty())  
        outfile.open(file.c_str(), std::ofstream::binary);

    bool overflow_message = true;

  /*  fftwf_plan plan = fftwf_plan_dft_1d(int(samps_per_buff),
        reinterpret_cast<fftwf_complex*>(buff.data()),
        reinterpret_cast<fftwf_complex*>(buff2.data()),
        FFTW_FORWARD, 
        FFTW_PATIENT); //or  FFTW_PATIENT //FFTW_MEASURE*/

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

         for (unsigned i = 0 ; i < 1024; ++i)
          buff3[i] = (buff2[i].real()*buff2[i].real() +buff2[i].imag()*buff2[i].imag())/16.0f;            

        for (unsigned i = 16; i < 1024; i++)
        {
            tmp[0] = buff3[i-15]+buff3[i-14]+buff3[i-13]+buff3[i-12];
            tmp[1] = buff3[i-11]+buff3[i-10]+buff3[i-9]+buff3[i-8];
            tmp[2] = buff3[i-7]+buff3[i-6]+buff3[i-5]+buff3[i-4];
            tmp[3] = buff3[i-3]+buff3[i-2]+buff3[i-1]+buff3[i];
            buff4[i]= (tmp[0]+tmp[1]+tmp[2]+tmp[3]);
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
    myfile.open("../cppL2.txt");

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
