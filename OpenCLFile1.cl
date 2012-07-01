//OpenCL Demo application made by Dmytro Konobrytskyi 2012

#include "MathCode.cpp"

__kernel void TestKernel(
	__global const float* pInputVector1, 
	__global const float* pInputVector2, 
	__global float* pOutputVectorHost, 
	int elementsNumber)
{
    //Get index into global data array
    int iJob = get_global_id(0);

    //Check boundary conditions
    if (iJob >= elementsNumber) return; 

    //Perform calculations
    pOutputVectorHost[iJob] = MathCalculations(pInputVector1[iJob], pInputVector2[iJob]);
}
