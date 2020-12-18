using AbstractSDRs
using DSP
using FFTW
using LinearAlgebra
using LoopVectorization
using BenchmarkTools
using StaticArrays
 """
Calculate rate based on Julia timing
main("uhd",10)
"""
function getRate(tInit,tFinal,nbSamples)
	return nbSamples / (tFinal-tInit);
end


function benchmark(samplingRate,duration)
#radioRx         = radio.rx;
	# --- Print the configuration

	nbSamples		= 1024;
	sig		  = zeros(Complex{Cfloat},nbSamples);
	abs_out	  = @MVector zeros(Cfloat,nbSamples); #@MVector
	mean_out  = @MVector zeros(Cfloat,nbSamples); #@MVector

	planFFT   = plan_fft(sig;flags=FFTW.PATIENT);
	internal  = similar(sig);
	temp1=0
	temp2=0

	# Init counter increment
	nS		  = 0;
	# Max counter definition
	nbBuffer  = duration*samplingRate;

	# --- Timestamp init
	timeInit  	= time();
	timeFinal   = timeInit+duration;
	while true


		abs_out .= abs2.(mul!(internal,planFFT,sig));

		 @inbounds for i in 16:nbSamples
			#mean_out[i] = sum(@views abs_out[i-15:i])/16.0
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
	effectiveRate = getRate(timeInit,timeFinal,nS);
    return (samplingRate,effectiveRate)
end

function main(sdr,duration;args="")

    # Init rate system
	desiredRate =[200e6];

    # File
    myFile = open("./juliaL2.txt", "w")

	montecarlo_num = 20
	try
		res = 0
	    # Benchmark
	    for r in desiredRate
			montecarlo_sum = 0
			 # Get the rate

			for montecarlo in 1:montecarlo_num
	        	(res,o) = benchmark(r,duration);
				@show (o/1e6)
				montecarlo_sum = montecarlo_sum + o/(montecarlo_num)
			end

			@show (res/1e6,montecarlo_sum/1e6)
	        # --- Write in file
	        write(myFile,"$(res) \t $(montecarlo_sum) \n");
	    end
	    close(myFile);

	catch y
		@show y
		close(myFile);
	end

end


main("uhd",10)
