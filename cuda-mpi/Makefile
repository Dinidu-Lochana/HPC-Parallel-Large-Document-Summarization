NVCC = nvcc

# Hardcoded MPICH paths for WSL to avoid nvcc flag parsing errors
MPI_CFLAGS = -I/usr/lib/x86_64-linux-gnu/mpich/include
MPI_LDFLAGS = -L/usr/lib/x86_64-linux-gnu/mpich/lib -lmpich

TARGET = cuda_mpi_summarizer
SRC = cuda_mpi_summarizer.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) -o $(TARGET) $(MPI_CFLAGS) $(MPI_LDFLAGS)

clean:
	rm -f $(TARGET) temp_hybrid_*
