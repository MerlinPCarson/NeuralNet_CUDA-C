NVCC=nvcc
EXE=mnist_test
SOURCE=main.cpp ../kernels.cu ../helpers.cpp ../data.cpp
INC=-I ../

all: 
	$(NVCC) $(INC) -std=c++11 -x cu -o $(EXE) $(SOURCE) 
clean:
	rm -rf $(EXE) 
