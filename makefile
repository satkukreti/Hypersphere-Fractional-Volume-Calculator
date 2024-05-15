all: ball_sam-cpu
	./ball_sam-cpu
	py graph.py

ball_sam-cpu: ball_sam-cpu.cpp
	g++ -O3 -o ball_sam-cpu ball_sam-cpu.cpp

clean:
	rm ball_sam-cpu
	rm output.txt