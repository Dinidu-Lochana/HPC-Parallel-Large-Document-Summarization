NVCC = nvcc
CFLAGS = -lpthread

TARGET = cuda_summarizer
SRC = cuda_summarizer.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) -o $(TARGET) $(CFLAGS)

clean:
	rm -f $(TARGET) temp_cuda_*
