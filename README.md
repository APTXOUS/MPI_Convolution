# MPI_Convolution
Use MPI for a BMP image convolution calculation. And a pthread version, too.

# Why we write it
What we want to do is to make the process of convolution as fast as possible, so most of the codes do not fit the common format.   
You can rewrite it for your purpose.


# How to use it
For mpi version:

    mpicxx con_mpi.cpp -o con_mpi
    mpirun -np 4 ./con_mpi

For pthread version

    g++ con_pthread.cpp -o con_pth -pthread
    ./con_pth 4

