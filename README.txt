To run this code:
1) For cpu:
    make

    It will output its result in a textfile called output.txt. It may take up to 2 minutes.
    It is formatted as dimension# followed by 100 probabilites from internval 1 to 100.
    It will automatically run ball_sam-cpu and run graph.py to plot the 3D Surface Plot.

2) For cuda
    I did not add to the makefile as it was tested on OpenHPC. Use OpenHPC to compile and run the code.