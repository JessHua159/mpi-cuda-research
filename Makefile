matrix-serial: matrix.c
	gcc -g matrix.c -o a.out

matrix-mpi: matrix-mpi.c
	mpicc -g matrix-mpi.c -o matrix-mpi

matrix-mpi-cuda: matrix-mpi-cuda.c matrix-mpi-cuda.cu
	mpixlc -g matrix-mpi-cuda.c -c -o matrix-mpi-cuda-xlc.o
	nvcc -g -G -c matrix-mpi-cuda.cu -o matrix-mpi-cuda.o
	mpicc -g matrix-mpi-cuda-xlc.o matrix-mpi-cuda.o -o matrix-mpi-cuda-exe \
	-L/usr/local/cuda-11.2/lib64/ \
	-lcudadevrt -lcudart -lcublas -lcublasLt -lcublas_static -lstdc++ \
	-lcudadevrt -lcudart -lcublas -lcublasLt -lcublas_static -lstdc++ \
	-lcudadevrt -lcudart -lcublas -lcublasLt -lcublas_static -lstdc++ \
	-lcudadevrt -lcudart -lcublas -lcublasLt -lcublas_static -lstdc++ \
	-lcudadevrt -lcudart -lcublas -lcublasLt -lcublas_static -lstdc++

all: matrix-serial matrix-mpi matrix-mpi-cuda
