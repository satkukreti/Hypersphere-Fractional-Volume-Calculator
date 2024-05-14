all: ball_sam-cuda

ball_sam-cuda: ball_sam-cuda.cu
	nvcc -O3 -o $@ $<

ball_sam-cpu: ball_sam-cpu.cpp
	g++ -O3 -o ball_sam-cpu ball_sam-cpu.cpp

clean:
	rm ball_sam-cpu
	rm ball_sam-cuda