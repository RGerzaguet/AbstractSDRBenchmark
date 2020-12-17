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
const double recv_to_process(uhd::rx_streamer::sptr rx_stream,
    const std::string& file,
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
    std::vector<samp_type> buff(samps_per_buff);
    std::vector<samp_type> buff2(samps_per_buff);
    std::vector<float> buff3(samps_per_buff);
    std::vector<float> buff4(samps_per_buff);

    std::complex<float>* out  = new std::complex<float>[samps_per_buff];

    std::ofstream outfile;
    if (not file.empty())  
        outfile.open(file.c_str(), std::ofstream::binary);

    bool overflow_message = true;

    fftwf_plan plan = fftwf_plan_dft_1d(int(samps_per_buff),
        reinterpret_cast<fftwf_complex*>(buff.data()),
        reinterpret_cast<fftwf_complex*>(buff2.data()),
        FFTW_FORWARD, 
        FFTW_PATIENT); //or  FFTW_PATIENT //FFTW_MEASURE


    // setup streaming
    uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
    stream_cmd.num_samps  = size_t(1024);
    stream_cmd.stream_now = true;
    stream_cmd.time_spec  = uhd::time_spec_t();
   // std::cout << "Issuing stream cmd" << std::endl;
    rx_stream->issue_stream_cmd(stream_cmd);

    const auto start_time = std::chrono::steady_clock::now();
    const auto stop_time = start_time + std::chrono::milliseconds(int64_t(1000 * time_requested));
    // Track time and samps between updating the BW summary
    auto last_update                     = start_time;
    unsigned long long last_update_samps = 0;

        // Run this loop until either time expired (if a duration was given), until
    // the requested number of samples were collected (if such a number was
    // given), or until Ctrl-C was pressed.
    while (not stop_signal_called and (time_requested == 0.0 or std::chrono::steady_clock::now() <= stop_time)) 
    {
        const auto now = std::chrono::steady_clock::now();

        num_total_samps += rx_stream->recv(&buff.front(), buff.size(), md, 3.0, enable_size_map);


        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) 
        {
            std::cout << boost::format("Timeout while streaming") << std::endl;
            break;
        }
        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) 
        {
            if (overflow_message) 
            {
                overflow_message = false;                
            }
            continue;
        }
        if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) 
        {
            std::string error = str(boost::format("Receiver error: %s") % md.strerror());
            if (continue_on_bad_packet) 
            {
                std::cerr << error << std::endl;
                continue;
            } 
            else
                throw std::runtime_error(error);
        }

    /////////////////////////////////////////////////////////////////////////
    /////////// Processing ////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////

        fftwf_execute(plan); 


        for (int i = 0 ; i < buff2.size(); ++i)
           buff3[i] = buff2[i].real()*buff2[i].real() +buff2[i].imag()*buff2[i].imag();            

      
        
 
    }
    const auto actual_stop_time = std::chrono::steady_clock::now();
    fftwf_destroy_plan(plan); 

    stream_cmd.stream_mode = uhd::stream_cmd_t::STREAM_MODE_STOP_CONTINUOUS;
    //std::cout << "Issuing stop stream cmd" << std::endl;
    rx_stream->issue_stream_cmd(stream_cmd);

    // Run recv until nothing is left
    int num_post_samps = 0;
    do {
        rx_stream->recv(&buff.front(), buff.size(), md, 3.0);
    } while (num_post_samps and md.error_code == uhd::rx_metadata_t::ERROR_CODE_NONE);
        std::cout << std::endl;


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
    setup_time = 2.0; //seconds of setup time   
    radio_chan = 0;//"radio channel to select inside ONE frontend
    radio_args = "";  // "Radio channel")
    rate = 100e6;// "RX rate of the radio block")
    freq=2450e6;//, "RF center frequency in Hz")
    gain = 10;// "gain for the RF chain")
    ant = "RX2"; //", po::value<std::string>(&ant), "antenna selection")
    bw = rate; //analog frontend filter bandwidth in Hz, !!! Will be overwritten in after DDC rate
    ref = "internal"; //reference source (internal, external, mimo)")
        
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
    /////////// Configure X310 //////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////
    std::cout << boost::format("Creating the USRP device with: %s...") % args  << std::endl;
    auto x310 = boost::dynamic_pointer_cast<uhd::device3>(uhd::device::make(args));


    /////////////////////////////////////////////////////////////////////////
    /////////// Configure Rx radio & DDC ////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////
    auto rx_radio_ctrl = x310->get_block_ctrl<radio_ctrl>(ID_RADIO0);
    std::cout << "-------------------------------------------------------------------------------------"<< std::endl;
    std::cout  << "Using radio " << ID_RADIO0 << ", channel " << radio_chan << std::endl;

    rx_radio_ctrl->set_clock_source(ref);
  
    // set the center frequency
    std::cout << boost::format("RX Freq \t Target: %f MHz \t-\t") % (freq / 1e6); 
    freq = rx_radio_ctrl->set_rx_frequency(freq, 0);
    std::cout << boost::format("Real: %f MHz") % (freq / 1e6) << std::endl;
    
    // set the rf gain
    std::cout << boost::format("RX Gain \t Target: %f MHz \t\t-\t") % gain;
    gain = rx_radio_ctrl->set_rx_gain(gain, radio_chan);
    std::cout << boost::format("Real: %f dB") % gain << std::endl;
 
    // set the antenna
    rx_radio_ctrl->set_rx_antenna(ant, radio_chan);

    // set rate
    if (not x310->has_block(ID_DDC)) {std::cout << "No DDC in device " << ID_DDC << std::endl; return EXIT_FAILURE; }
    auto ddc_ctrl = x310->get_block_ctrl<uhd::rfnoc::ddc_block_ctrl>(ID_DDC);        
    std::cout << boost::format("RX Rate \t Target: %f MHz \t-\t") % (rate / 1e6); 
    ddc_ctrl->set_arg("input_rate", 200e6,0); //change the 0 if multiple DDC
    ddc_ctrl->set_arg("output_rate", rate,0);
    ddc_ctrl->set_arg("fullscale", 1.0,0);
    ddc_ctrl->set_arg("freq", 0.0,0);    
    rate = ddc_ctrl->get_args().cast<float>("output_rate",-1e6);    
    std::cout << boost::format("Real: %f Msps") % (rate / 1e6) << std::endl;


    // set the IF filter bandwidth
    bw = rate;
    std::cout << boost::format("RX Bandwidth \t Target: %f MHz \t-\t") % (bw/1e6); 
    bw = rx_radio_ctrl->set_rx_bandwidth(bw, radio_chan); 
    std::cout << boost::format("Real: %f MHz")% (bw/1e6) << std::endl;

    // wait stabilize
    std::this_thread::sleep_for(std::chrono::milliseconds(int64_t(1000 * setup_time)));
    

    /////////////////////////////////////////////////////////////////////////
    /////////// Set up streaming & connect //////////////////
    /////////////////////////////////////////////////////////////////////////
    uhd::device_addr_t streamer_args(streamargs);
    uhd::rfnoc::graph::sptr rx_graph = x310->create_graph("tempest");
    x310->clear();
    

    // Connect:
    std::cout << "Connecting " << rx_radio_ctrl->get_block_id() << " ==> " << ddc_ctrl->get_block_id() << std::endl;
    rx_graph->connect(rx_radio_ctrl->get_block_id() , radio_chan, ddc_ctrl->get_block_id(), 0, spp);
    streamer_args["block_id"] = ddc_ctrl->get_block_id().to_string();

    // create a receive streamer
    std::cout << "Samples per packet: " << spp << std::endl;
    uhd::stream_args_t stream_args(format, "sc16"); // We should read the wire format from the blocks
    stream_args.args        = streamer_args;
    stream_args.args["spp"] = boost::lexical_cast<std::string>(spp);
    std::cout << "Using streamer args: " << stream_args.args.to_string() << std::endl;
    uhd::rx_streamer::sptr rx_stream = x310->get_rx_stream(stream_args);

    /////////////////////////////////////////////////////////////////////////
    /////////// Set up benchmark /////////////////////////////////
    /////////////////////////////////////////////////////////////////////////
    
    
    
   double rcv_rate;

    std::ofstream myfile;
    myfile.open("../../../data_benchmark_cpp_abs2_quick.txt");

    std::array<double,9> desiredRate = {2e6,10e6,20e6,30e6,40e6,50e6,75e6,100e6,200e6};
    int     processWait = 1;
    for (auto it = desiredRate.begin(); it != desiredRate.end(); ++it)
    {
        rate = *it;

        // set rate
        if (not x310->has_block(ID_DDC)) {std::cout << "No DDC in device " << ID_DDC << std::endl; return EXIT_FAILURE; }
        auto ddc_ctrl = x310->get_block_ctrl<uhd::rfnoc::ddc_block_ctrl>(ID_DDC);        
        std::cout << boost::format("RX Rate \t Target: %f MHz \t-\t") % (rate / 1e6); 
        ddc_ctrl->set_arg("input_rate", 200e6,0); //change the 0 if multiple DDC
        ddc_ctrl->set_arg("output_rate", rate,0);
        ddc_ctrl->set_arg("fullscale", 1.0,0);
        ddc_ctrl->set_arg("freq", 0.0,0);    
        rate = ddc_ctrl->get_args().cast<float>("output_rate",-1e6);    
        std::cout << boost::format("Real: %f Msps") % (rate / 1e6) << std::endl;

        // set the IF filter bandwidth
        bw = rate;
        bw = rx_radio_ctrl->set_rx_bandwidth(bw, radio_chan); 
      

        // wait stabilize
        std::this_thread::sleep_for(std::chrono::milliseconds(int64_t(1000 * setup_time)));

        #define recv_to_process_args() \
            (rx_stream,             \
                file,               \
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
