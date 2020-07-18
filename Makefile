NVCC=/usr/local/cuda-10.0/bin/nvcc
EXE=MNIST
SOURCE=main.cpp kernels.cu neural_net.cpp data.cpp helpers.cpp

all: 
	$(NVCC) -x cu -o $(EXE) $(SOURCE) 
clean:
	rm -rf $(EXE) 
