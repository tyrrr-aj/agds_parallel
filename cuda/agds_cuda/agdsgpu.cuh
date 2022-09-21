#pragma once

#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include "avbtree.hpp"
#include "measurements.hpp"

#include <vector>
#include <iostream>
#include <string>

#define __FILENAME__ (std::string(__FILE__).substr(std::string(__FILE__).find_last_of("/\\") + 1))

// the following two macros CHECK_CUDA adn CHECKCUSPARSE are taken from NVIDA sample (and modified):
// https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/dense2sparse_blockedell/dense2sparse_blockedell_example.c

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        std::cout << "CUDA API failed at line "								   \
				  << __LINE__												   \
				  << " (" << __FILENAME__ << ")"							   \
				  << " with error: "										   \
				  << cudaGetErrorString(status)								   \
				  << " (" << status << ")\n";								   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
		std::cout << "CUSPARSE API failed at line "							   \
				  << __LINE__												   \
				  << " (" << __FILENAME__ << ")"							   \
				  << " with error: "										   \
				  << cusparseGetErrorString(status)							   \
				  << " (" << status << ")\n";								   \
    }                                                                          \
}


#define CHECK_CUBLAS(func)                                                   \
{                                                                              \
    cublasStatus_t status = (func);                                          \
    if (status != CUBLAS_STATUS_SUCCESS) {                                   \
		std::cout << "CUBLAS API failed at line "							   \
				  << __LINE__												   \
				  << " (" << __FILENAME__ << ")"							   \
				  << " with error: "										   \
				  << cublasGetStatusString(status)							   \
				  << " (" << status << ")\n";								   \
    }                                                                          \
}


#define CUDA_DEFAULT_STREAM 0


template<typename T>
__global__ void insertAndShift(T* vec, T newValue, int newValueIdx, int vecLen) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	if (x <= vecLen) {
		if (x == newValueIdx) {
			vec[x] = newValue;
		}
		else if (x > newValueIdx) {
			vec[x] = vec[x - 1];
		}
	}
}


template<typename T>
class VecDense {
public:
	int len;
	int reservedSpace = 1024;
	T* valuesDev;

	void append(T newValue) {
		append(&newValue, 1);
	}

	void append(T* values, int nValues) {
		while (len + nValues >= reservedSpace) {
			realloc();
		}

		CHECK_CUDA(cudaEventSynchronize(bufferReadyToOverwrite));
		CHECK_CUDA(cudaEventDestroy(bufferReadyToOverwrite));

		for (int i = 0; i < nValues; i++) {
			buffer[i] = values[i];
		}

		CHECK_CUDA(cudaEventCreate(&bufferReadyToOverwrite, cudaEventBlockingSync));

		CHECK_CUDA(cudaMemcpyAsync(valuesDev + len, buffer, nValues * sizeof(T), cudaMemcpyHostToDevice, stream));
		CHECK_CUDA(cudaEventRecord(bufferReadyToOverwrite, stream));
		len += nValues;
		int a = 1;
	}

	void set(int index, const T& newValue) {
		if (index >= len) {
			append(newValue);
		}
		else {
			CHECK_CUDA(cudaMemcpyAsync(valuesDev + index, &newValue, sizeof(T), cudaMemcpyHostToDevice, stream));
		}
	}

	void insert(int index, const T& newValue) {
		if (index >= len) {
			append(newValue);
		}
		else {
			const int threadsPerBlock = 256;
			int nBlocks = ((len + 1) + threadsPerBlock) / threadsPerBlock;
			len++;

			if (len >= reservedSpace) {
				realloc();
			}

			insertAndShift<T> << <nBlocks, threadsPerBlock, 0, stream >> > (valuesDev, newValue, index, len);
		}
	}

	VecDense(cudaStream_t stream = 0) {
		len = 0;
		this->stream = stream;
		CHECK_CUDA(cudaMallocAsync(&valuesDev, reservedSpace * sizeof(T), stream));
		CHECK_CUDA(cudaHostAlloc(&buffer, 1 * sizeof(T), cudaHostAllocWriteCombined));
		CHECK_CUDA(cudaEventCreate(&bufferReadyToOverwrite, cudaEventBlockingSync));
		CHECK_CUDA(cudaEventRecord(bufferReadyToOverwrite, stream));
	}

	~VecDense() {
		CHECK_CUDA(cudaFreeAsync(valuesDev, stream));
		CHECK_CUDA(cudaFreeHost(buffer));
		CHECK_CUDA(cudaEventDestroy(bufferReadyToOverwrite));
	}

	void print(std::string name) {
		T* valuesHost = new T[len];
		CHECK_CUDA(cudaMemcpy(valuesHost, valuesDev, len * sizeof(T), cudaMemcpyDeviceToHost));

		std::cout << name << ": ";

		for (int i = 0; i < len; i++) {
			std::cout << valuesHost[i] << " ";
		}
		std::cout << std::endl;

		delete[] valuesHost;
	}

private:
	void realloc() {
		reservedSpace *= 2;
		T* tmp;
		CHECK_CUDA(cudaMallocAsync(&tmp, reservedSpace * sizeof(T), 0));
		CHECK_CUDA(cudaMemcpyAsync(tmp, valuesDev, len * sizeof(T), cudaMemcpyDeviceToDevice));
		CHECK_CUDA(cudaFreeAsync(valuesDev, 0));
		valuesDev = tmp;
	}

	cudaStream_t stream;
	T* buffer = NULL;
	cudaEvent_t bufferReadyToOverwrite = 0;
};


class CONN {
public:
	VecDense<float> cooNzValuesDev;
	VecDense<int> cooRowIndicesDev;
	VecDense<int> cooColIndicesDev;

	int nOn = 0;
	int nVn = 0;

	void addON(int vnOffset);
	cusparseSpMatDescr_t getDescr();

	CONN();
	~CONN();

private:
	cusparseSpMatDescr_t descr;
	bool isDescrReady = false;

	cudaStream_t stream = 0;

	void freeDescr();
};


class VNG {
	AVBTree<float> tree;

	void updateP(TreeInsertResult<float>& insertRes, float newValue);
	void updateNrev(TreeInsertResult<float>& insertRes);
	void updateCONN(TreeInsertResult<float>& insertRes);

	cusparseDnVecDescr_t AVnDescr = 0;
	bool AVnDescrReady = false;

	void freeAVnDescr();

public:
	VecDense<float>* P;
	VecDense<int>* Indices;
	VecDense<float>* Nrev;

	float* AVn;
	float* AVnDev;

	CONN Conn;

	void addValue(float newValue);
	int getNVn();

	cusparseDnVecDescr_t getAVnDescr();
	void resetAVn();
	void loadAVnToHost();
	void printAVn();

	cudaStream_t stream;
	cudaEvent_t vngReadyForVn2OnEvent;

	VNG();
	~VNG();
};


class AGDS {
public:
	std::vector<VNG> vngs;
	int n_on = 0;
	int n_vng;
	int n_vn();

	float* AOn = NULL;

	void addObservation(float* values);

	void setupOnQuery(int* activatedOns, int nActivatedOns);
	void infere();

	void printAOn();
	void loadResultsToHost();

	AGDS(float* data, int n_on, int n_vng);
	~AGDS();

private:
	cusparseHandle_t cusparseHandle;
	cublasHandle_t cublasHandle;

	cusparseDnVecDescr_t AOnDescr;
	float* AOnDev = NULL;
	bool isAOnReady = false;
	void freeAOnDescr();

	void on2vn();
	void vn2vn();
	void vn2on();

	Measurer measurer;
};
