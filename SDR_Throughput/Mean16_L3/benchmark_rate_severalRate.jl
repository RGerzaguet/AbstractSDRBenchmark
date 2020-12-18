using AbstractSDRs
using DSP
using FFTW
using LinearAlgebra
using LoopVectorization
using StaticArrays
 """
Calculate rate based on Julia timing
main("uhd",10)
"""
function getRate(tInit,tFinal,nbSamples)
	return nbSamples / (tFinal-tInit);
end


function benchmark(radio,samplingRate,duration)
    radioRx         = radio.rx;
	# --- Print the configuration

	#print(radioRx);
	# --- Init parameters
	# Get the radio size for buffer pre-allocation
	nbSamples		= 1024;
	# We will get complex samples from recv! method
	sig		  = zeros(Complex{Cfloat},nbSamples);
	abs_out	  = @MVector zeros(Cfloat,nbSamples);
	mean_out  = @MVector zeros(Cfloat,nbSamples);

	planFFT   = plan_fft(sig;flags=FFTW.PATIENT);
	internal  = similar(sig);

	# Init counter increment
	nS		  = 0;
	# Max counter definition
	nbBuffer  = duration*samplingRate;

	# --- Timestamp init
	timeInit  	= time();
	timeFinal   = timeInit+duration;
	while true
		# --- Direct call to avoid allocation
		recv!(sig,radio);

        mul!(internal,planFFT,sig)
		@inbounds @avx for i in 1:1:nbSamples		
 			abs_out[i] = abs2(internal[i])
		end

		@inbounds @avx for i in 16:nbSamples
			mean_out[i] = (abs_out[i]+abs_out[i-1]+abs_out[i-2]+abs_out[i-3]+abs_out[i-4]+abs_out[i-5]+abs_out[i-6]+abs_out[i-7]+abs_out[i-8]+abs_out[i-9]+abs_out[i-10]+abs_out[i-11]+abs_out[i-12]+abs_out[i-13]+abs_out[i-14]+abs_out[i-15])/16.0
		end

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
    myFile = open("data_benchmark_julia_mean16_L3.txt", "w")

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
		close(myFile);
	    @show close(radio);
	end

end
main("uhd",10)
