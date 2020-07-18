#include <stdio.h>
#include <stdlib.h>

#include "kernels.h"
#include "neural_net.h"
#include "data.h"
#include "helpers.h"


int main(int argc, char * argv[])
{

  // identify cuda devices
  if(!cudaDeviceProperties()){
    return 1;
  }

  return 0;
}
