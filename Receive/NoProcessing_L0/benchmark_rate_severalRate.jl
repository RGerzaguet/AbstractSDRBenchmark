using AbstractSDRs
using DSP
using FFTW
using LinearAlgebra
 """
Calculate rate based on Julia timing
main("uhd",10)
"""
function getRate(tInit,tFinal,nbSamples)
	return nbSamples / (tFinal-tInit);
end


function benchmark(radio,samplingRate,duration)
    radioRx         = radio.rx;

	# --- Init parameters
	# Get the radio size for buffer pre-allocation
	nbSamples		= 1024;
	mean_out  = zeros(Float32,nbSamples);

	# Init counter increment
	nS		  = 0;
	# Max counter definition
	nbBuffer  = duration*samplingRate;
	# --- Timestamp init

	timeInit  	= time();
	timeFinal   = timeInit+duration;
	while true
		sig = recv(radio,nbSamples)

		nS		+= nbSamples;
		# --- Interruption
		if time() > timeFinal
			# Last timeStamp
            timeFinal = time();
			break
		end
	end
	# --- Getting effective rate
	radioRate	  = radioRx.samplingRate;
    effectiveRate = getRate(timeInit,timeFinal,nS);
    return (radioRate,effectiveRate)
end

function main(sdr,duration;args="")

    # Create radio
    carrierFreq  = 868e6;
    samplingRate = 1e6;
    gain         = 10;
    radio        = openSDR(sdr,carrierFreq,samplingRate,gain;args=args);
    # Check everything is OK
    print(radio);

    # Init rate system
	desiredRate =[2e6,10e6,20e6,30e6,40e6,50e6,75e6,100e6,200e6];

    # File
    myFile = open("data_benchmark_julia_noProcessing_L0.txt", "w")

	montecarlo_num = 20
	try
		res = 0
	    # Benchmark
	    for r in desiredRate
			montecarlo_sum = 0
			 # Get the rate
			updateSamplingRate!(radio,r);

			for montecarlo in 1:montecarlo_num
	        	(res,o) = benchmark(radio,r,duration);
				@show (o/1e6)
				montecarlo_sum = montecarlo_sum + o/(montecarlo_num)
			end

			@show (res/1e6,montecarlo_sum/1e6)
	        # --- Write in file
	        write(myFile,"$(res) \t $(montecarlo_sum) \n");
	    end
	    close(myFile);
	    close(radio);
	catch y
		@info y
		close(myFile);
	    close(radio);
	end

end
main("uhd",10)
