//OpenCL Demo application made by Dmytro Konobrytskyi 2012

#pragma comment( lib, "opencl.lib" )
#define __CL_ENABLE_EXCEPTIONS
#include "cl.hpp"

#include <time.h>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <numeric>
#include <iomanip>
#include <algorithm>
#include <conio.h>

#include <thread>
#include <ppl.h>
#include <array>
#include <sstream>
#include <iostream>
#include <vector>

using namespace Concurrency;
using namespace std;

float MathCalculations(float a, float b);

const int DATA_SIZE = 20*1024*1024;
const int TESTS_NUMBER = 200;
float *pInputVector1;
float *pInputVector2;
float *pOutputVector;
float *pOutputVectorHost;
double hostPerformanceTimeMS = 0;
std::vector<double> timeValues;

void PrintTimeStatistic()
{
	std::sort(timeValues.begin(), timeValues.end());
	double totalTime = std::accumulate(timeValues.begin(), timeValues.end(), 0.0);
	double averageTime = totalTime/timeValues.size();
	double minTime = timeValues[0];
	double maxTime = timeValues[timeValues.size()-1];
	double medianTime = timeValues[timeValues.size()/2];
	cout << "Calculation time statistic: (" << timeValues.size() << " runs)" << endl;
	cout << "Med: " << medianTime << " ms (" << hostPerformanceTimeMS/medianTime << "X faster then host)" << endl;
	cout << "Avg: " << averageTime << " ms" << endl;
	cout << "Min: " << minTime << " ms" << endl;
	cout << "Max: " << maxTime << " ms" << endl << endl;
}

void GenerateTestData()
{
	pInputVector1 = new float[DATA_SIZE];
	pInputVector2 = new float[DATA_SIZE];
	pOutputVector = new float[DATA_SIZE];
	pOutputVectorHost = new float[DATA_SIZE];

	srand ((unsigned int)time(NULL));
	for (int i=0; i<DATA_SIZE; i++)
	{
		pInputVector1[i] = rand() * 1000.0f / RAND_MAX;
		pInputVector2[i] = rand() * 1000.0f / RAND_MAX;
	}
}

void STDThreadCalculationFunction(int start, int end)
{
	for(int iJob=start; iJob<end; iJob++)
	{
		//Perform calculations
		pOutputVector[iJob] = MathCalculations(pInputVector1[iJob], pInputVector2[iJob]);
	}
}

void PerformCalculationsOnHostSeparateFunction()
{
    cout << endl << "-------------------------------------------------" << endl;
    cout << "Device: Host separate function" << endl << endl;

    //Some performance measurement
    timeValues.clear();
    __int64 start_count;
    __int64 end_count;
    __int64 freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

    for(int iTest = 0; iTest < (TESTS_NUMBER / 5); iTest++)
    {
        QueryPerformanceCounter((LARGE_INTEGER*)&start_count);
        STDThreadCalculationFunction(0, DATA_SIZE);
        QueryPerformanceCounter((LARGE_INTEGER*)&end_count);
        double time = 1000 * (double)(end_count - start_count) / (double)freq;
        timeValues.push_back(time);
    }

    PrintTimeStatistic();
}

void PerformCalculationsOnHost()
{
	cout << "Device: Host" << endl << endl;

	//Some performance measurement
	timeValues.clear();
	__int64 start_count;
	__int64 end_count;
	__int64 freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	for(int iTest=0; iTest<(TESTS_NUMBER/10); iTest++)
	{
		QueryPerformanceCounter((LARGE_INTEGER*)&start_count);
		for(int iJob=0; iJob<DATA_SIZE; iJob++)
		{
			//Perform calculations
			pOutputVectorHost[iJob] = MathCalculations(pInputVector1[iJob], pInputVector2[iJob]);
		}
		QueryPerformanceCounter((LARGE_INTEGER*)&end_count);
		double time = 1000 * (double)(end_count - start_count) / (double)freq;
		timeValues.push_back(time);
	}
	hostPerformanceTimeMS = std::accumulate(timeValues.begin(), timeValues.end(), 0.0)/timeValues.size();

	PrintTimeStatistic();
}

void PerformCalculationsOnHostParallelFor()
{
	cout << endl << "-------------------------------------------------" << endl;
	cout << "Device: Host parallel_for" << endl << endl;

	//Some performance measurement
	timeValues.clear();
	__int64 start_count;
	__int64 end_count;
	__int64 freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	for(int iTest=0; iTest<(TESTS_NUMBER/5); iTest++)
	{
		QueryPerformanceCounter((LARGE_INTEGER*)&start_count);
		parallel_for(size_t(0), size_t(DATA_SIZE), [&](size_t iJob)
		{
			//Perform calculations
			pOutputVector[iJob] = MathCalculations(pInputVector1[iJob], pInputVector2[iJob]);
		});
		QueryPerformanceCounter((LARGE_INTEGER*)&end_count);
		double time = 1000 * (double)(end_count - start_count) / (double)freq;
		timeValues.push_back(time);
	}

	PrintTimeStatistic();
}

void PerformCalculationsOnHostSTDThread()
{
	cout << endl << "-------------------------------------------------" << endl;
	cout << "Device: Host std::thread" << endl << endl;

	//Some performance measurement
	timeValues.clear();
	__int64 start_count;
	__int64 end_count;
	__int64 freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	int threadsNumber = max(1, std::thread::hardware_concurrency());
	cout << "Threads number: " << threadsNumber << endl << endl;
	int jobsPerThread = DATA_SIZE/threadsNumber;

	for(int iTest=0; iTest<(TESTS_NUMBER/5); iTest++)
	{
		QueryPerformanceCounter((LARGE_INTEGER*)&start_count);
		
		int curStartJob = 0;
		std::vector<std::thread> threadVector;
		for(int iThread=0; iThread<threadsNumber; iThread++)
		{
			threadVector.push_back(std::thread(STDThreadCalculationFunction, curStartJob, min(curStartJob+jobsPerThread, DATA_SIZE)));
			curStartJob += jobsPerThread;
		}

		for(auto thread=threadVector.begin(); thread!=threadVector.end(); thread++)
			thread->join();

		QueryPerformanceCounter((LARGE_INTEGER*)&end_count);
		double time = 1000 * (double)(end_count - start_count) / (double)freq;
		timeValues.push_back(time);
	}

	PrintTimeStatistic();
}

void PerformCalculationsOnHostSTDThread1()
{
	cout << endl << "-------------------------------------------------" << endl;
	cout << "Device: Host std::thread 1 " << endl << endl;

	//Some performance measurement
	timeValues.clear();
	__int64 start_count;
	__int64 end_count;
	__int64 freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	int threadsNumber = 1;
	cout << "Threads number: " << threadsNumber << endl << endl;
	int jobsPerThread = DATA_SIZE/threadsNumber;

	for(int iTest=0; iTest<(TESTS_NUMBER/5); iTest++)
	{
		QueryPerformanceCounter((LARGE_INTEGER*)&start_count);
		
		int curStartJob = 0;
		std::vector<std::thread> threadVector;
		for(int iThread=0; iThread<threadsNumber; iThread++)
		{
			threadVector.push_back(std::thread(STDThreadCalculationFunction, curStartJob, min(curStartJob+jobsPerThread, DATA_SIZE)));
			curStartJob += jobsPerThread;
		}

		for(auto thread=threadVector.begin(); thread!=threadVector.end(); thread++)
			thread->join();

		QueryPerformanceCounter((LARGE_INTEGER*)&end_count);
		double time = 1000 * (double)(end_count - start_count) / (double)freq;
		timeValues.push_back(time);
	}

	PrintTimeStatistic();
}

void PerformTestOnDevice(cl::Device device)
{
	cout << endl << "-------------------------------------------------" << endl;
	cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << endl << endl;

	//For the selected device create a context
	vector<cl::Device> contextDevices;
	contextDevices.push_back(device);
	cl::Context context(contextDevices);

	//For the selected device create a context and command queue
	cl::CommandQueue queue(context, device);

	//Clean output buffers
	fill_n(pOutputVector, DATA_SIZE, 0);

	//Create memory buffers
	cl::Buffer clmInputVector1 = cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), pInputVector1);
	cl::Buffer clmInputVector2 = cl::Buffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), pInputVector2);
	cl::Buffer clmOutputVector = cl::Buffer(context, CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), pOutputVector);

	//Load OpenCL source code
	std::ifstream sourceFile("OpenCLFile1.cl");
	std::string sourceCode(std::istreambuf_iterator<char>(sourceFile),(std::istreambuf_iterator<char>()));

	//Build OpenCL program and make the kernel
	cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
	cl::Program program = cl::Program(context, source);
	program.build(contextDevices);
	cl::Kernel kernel(program, "TestKernel");

	//Set arguments to kernel
	int iArg = 0;
	kernel.setArg(iArg++, clmInputVector1);
	kernel.setArg(iArg++, clmInputVector2);
	kernel.setArg(iArg++, clmOutputVector);
	kernel.setArg(iArg++, DATA_SIZE);

	//Some performance measurement
	timeValues.clear();
	__int64 start_count;
	__int64 end_count;
	__int64 freq;
	QueryPerformanceFrequency((LARGE_INTEGER*)&freq);

	//Run the kernel on specific ND range
	for(int iTest=0; iTest<TESTS_NUMBER; iTest++)
	{
		QueryPerformanceCounter((LARGE_INTEGER*)&start_count);

		queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(DATA_SIZE), cl::NDRange(128));
		queue.finish();

		QueryPerformanceCounter((LARGE_INTEGER*)&end_count);
		double time = 1000 * (double)(end_count - start_count) / (double)freq;
		timeValues.push_back(time);
	}

	PrintTimeStatistic();

	// Read buffer C into a local list
	queue.enqueueReadBuffer(clmOutputVector, CL_TRUE, 0, DATA_SIZE * sizeof(float), pOutputVector);
}

void CheckResults()
{
	double avgRelAbsDiff = 0;
	double maxRelAbsDiff = 0;
	for(int iJob=0; iJob<DATA_SIZE; iJob++)
	{
		double absDif = abs(pOutputVectorHost[iJob] - pOutputVector[iJob]);
		double relAbsDif = abs(absDif/pOutputVectorHost[iJob]);
		avgRelAbsDiff += relAbsDif;
		maxRelAbsDiff = max(maxRelAbsDiff, relAbsDif);
		pOutputVector[iJob] = 0;
	}
	avgRelAbsDiff /= DATA_SIZE;

	cout << "Errors:" << endl;
	cout << "avgRelAbsDiff = " << avgRelAbsDiff << endl;
	cout << "maxRelAbsDiff = " << maxRelAbsDiff << endl;
}

int main(int argc, char* argv[])
{
	GenerateTestData();
	PerformCalculationsOnHost();
    PerformCalculationsOnHostSeparateFunction();
    CheckResults();
    PerformCalculationsOnHostParallelFor();
	CheckResults();
	PerformCalculationsOnHostSTDThread();
	CheckResults();
	PerformCalculationsOnHostSTDThread1();
	CheckResults();

	//Get all available platforms
	vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	for (unsigned int iPlatform=0; iPlatform<platforms.size(); iPlatform++)
	{
		//Get all available devices on selected platform
		std::vector<cl::Device> devices;
		platforms[iPlatform].getDevices(CL_DEVICE_TYPE_ALL, &devices);

		//Perform test on each device
		for (unsigned int iDevice=0; iDevice<devices.size(); iDevice++)
		{
			try 
			{ 
				PerformTestOnDevice(devices[iDevice]);
			} 
			catch(cl::Error error) 
			{
				std::cout << error.what() << "(" << error.err() << ")" << std::endl;
			}
			CheckResults();
		}
	}

	//Clean buffers
	delete[](pInputVector1);
	delete[](pInputVector2);
	delete[](pOutputVector);
	delete[](pOutputVectorHost);

	_getch();

	return 0;
}

