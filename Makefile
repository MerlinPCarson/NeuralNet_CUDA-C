#NVCC=/usr/local/cuda-10.0/bin/nvcc
#NVCC=/pkgs/nvidia-cuda/10.0/bin/nvcc
NVCC=nvcc
EXE=MNIST
SOURCE=main.cpp kernels.cu neural_net.cpp data.cpp helpers.cpp

all: 
	$(NVCC) -std=c++11 -x cu -o $(EXE) $(SOURCE) 
clean:
	rm -rf $(EXE) 
