# Hypersphere Fractional Volume Calculator

This is the final project for my High Performance Computing class. I executed the CUDA code on Binghamton's Computing Cluster using OpenHPC and the Slurm Workload Manager. Basic profiling was done using gperftools to optimize (not very well (-｡-;) the code. Saw a ~14x speedup.

## Simulation Goal

This program estimates the function \( f_d(l) \), where \( f_d(l) \) is the fraction of the volume of a **D-dimensional** hypersphere (read: sphere whose dimensions are based on surface area, i.e., an ordinary sphere is a 2-sphere as the surface area is 2-dimensional) that is within distance \( l \) of the surface. The distance \( l \) is calculated from 0 to 1 in steps of 0.01.

I used a random sampling of uniformly distributed points to simulate the sphere. Points outside the sphere were rejected by calculating the distance from the center, leaving the rest to approximate the fractional volume.

## To Run This Code

### For CPU:

1. Compile the code:
    ```bash
    make
    ```

2. The result will be output to a text file called `output.txt`. It may take up to 2 minutes to complete. The file is formatted as dimension# followed by 100 probabilities from interval 1 to 100.

3. The code will automatically run `ball_sam-cpu` and `graph.py` to plot the 3D Surface Plot.

### For CUDA:

I did not add the CUDA code to the makefile as it was tested on OpenHPC. Use OpenHPC and `nvcc` to compile and run the code.

---

### Example Usage for CUDA

To compile and run the CUDA version, use the following commands on an OpenHPC setup:
```bash
module avail
module add cuda/12.0
which nvcc
nvcc my_code.cu -o a.out
srun -p gpu ./a.out
```
