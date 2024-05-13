all: ball_sam-cpu

ball_sam-cpu: ball_sam-cpu.cpp
	g++ ball_sam-cpu.cpp -o ball_sam-cpu

clean:
	rm ball_sam-cpu