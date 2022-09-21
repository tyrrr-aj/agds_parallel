#pragma once

#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <iostream>

#include "agds_components.hpp"


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


class ConnSparse {
	cusparseHandle_t cusparseHandle;

	float* nzValuesCOO;
	int* rowIndicesCOO;
	int* colIndicesCOO;

	float* nzValuesCOOCSRDev = 0;
	int* rowIndicesCOODev = 0;
	int* rowIndicesCSRDev = 0;
	int* colIndicesCOOCSRDev = 0;

	float* nzValuesCSRTransposedDev = 0;
	int* rowIndicesCSRTransposedDev = 0;
	int* colIndicesCSRTransposedDev = 0;

	void createCOO();
	void coo2csr();
	void transposeCSR();
	void csr2bsr(
		int nRows,
		int nCols,
		float* nzValuesCSRDev,
		int* rowIndicesCSRDev,
		int* colIndicesCSRDev,
		float*& nzValuesBSRDev,
		int*& rowIndicesBSRDev,
		int*& colIndicesBSRDev
	);

public:
	const cusparseDirection_t matrixDir = CUSPARSE_DIRECTION_ROW;
	const int myBlockDim = 3; // arbitrary, intuitive choice - may (should) be adjusted

	cusparseMatDescr_t descrGen = 0;
	cusparseSpMatDescr_t descrSparse = 0;

	int nnz;
	int nRows;
	int nCols;

	int getNBlocksRows();
	int getNBlocksCols();

	float* nzValuesBSRDev = 0;
	int* rowIndicesBSRDev = 0;
	int* colIndicesBSRDev = 0;

	float* nzValuesBSRTransposedDev = 0;
	int* rowIndicesBSRTransposedDev = 0;
	int* colIndicesBSRTransposedDev = 0;

	void setup();

	ConnSparse(Conn* const conn, cusparseHandle_t cusparseHandle);
	~ConnSparse();
};


class VecDense {
	cusparseHandle_t cusparseHandle;

	bool deviceMemControl = false;

	void setupOnDevice(int lengthPadded) {
		deviceMemControl = true;
		// std::cout << "allocating: " << lengthPadded << " floats\n";
		CHECK_CUDA(cudaMalloc(&valuesDev, lengthPadded * sizeof(float)));
		CHECK_CUDA(cudaMemcpy(valuesDev, values, length * sizeof(float), cudaMemcpyHostToDevice));
	}

public:
	cusparseDnVecDescr_t descr = 0;
	int length;
	float* values;
	float* valuesDev;

	void setValue(int index, float value) {
		values[index] = value;
		// CHECK_CUDA(cudaMemset(valuesDev + index, value, sizeof(float)));
		CHECK_CUDA(cudaMemcpy(valuesDev + index, values + index, sizeof(float), cudaMemcpyHostToDevice));
	}

	void loadFromDevice() {
		CHECK_CUDA(cudaMemcpy(values, valuesDev, length * sizeof(float), cudaMemcpyDeviceToHost));
	}

	void loadToDevice() {
		CHECK_CUDA(cudaMemcpy(valuesDev, values, length * sizeof(float), cudaMemcpyHostToDevice));
	}

	VecDense(float* values, int length, int lengthPadded, cusparseHandle_t cusparseHandle) {
		this->cusparseHandle = cusparseHandle;
		this->values = values;
		this->length = length;

		setupOnDevice(lengthPadded);

		CHECK_CUSPARSE(cusparseCreateDnVec(&descr, lengthPadded, valuesDev, CUDA_R_32F));
	}

	VecDense(float* values, int length, cusparseHandle_t cusparseHandle) {
		this->cusparseHandle = cusparseHandle;
		this->values = values;
		this->length = length;

		setupOnDevice(length);

		CHECK_CUSPARSE(cusparseCreateDnVec(&descr, length, valuesDev, CUDA_R_32F));
	}

	VecDense(float* valuesOnHost, float* valuesOnDevice, int length, cusparseHandle_t cusparseHandle) {
		this->cusparseHandle = cusparseHandle;
		values = valuesOnHost;
		this->length = length;
		valuesDev = valuesOnDevice;
		CHECK_CUSPARSE(cusparseCreateDnVec(&descr, length, valuesDev, CUDA_R_32F));
	}

	VecDense(const float initValue, int length, cusparseHandle_t cusparseHandle) {
		this->cusparseHandle = cusparseHandle;
		this->length = length;

		values = new float[length];
		for (int i = 0; i < length; i++) {
			values[i] = initValue;
		}

		setupOnDevice(length);

		CHECK_CUSPARSE(cusparseCreateDnVec(&descr, length, valuesDev, CUDA_R_32F));
	}

	VecDense(const float initValue, int length, int lengthPadded, cusparseHandle_t cusparseHandle) {
		this->cusparseHandle = cusparseHandle;
		this->length = length;

		values = new float[length];
		for (int i = 0; i < length; i++) {
			values[i] = initValue;
		}

		setupOnDevice(lengthPadded);

		CHECK_CUSPARSE(cusparseCreateDnVec(&descr, lengthPadded, valuesDev, CUDA_R_32F));
	}

	~VecDense() {
		if (deviceMemControl) {
			CHECK_CUDA(cudaFree(valuesDev));
		}
		CHECK_CUSPARSE(cusparseDestroyDnVec(descr));
	}
};


class VnVecDense {
	cusparseHandle_t cusparseHandle;
	VnVec<float>* srcVec;
	bool isSrcVecCreatedInternally = false;

	void setup(VnVec<float>* srcVec, int fullLengthPadded, cusparseHandle_t cusparseHandle);

public:
	VecDense* fullVec;
	VecDense** vecPartsVngs;

	void loadFromDevice();
	void loadToDevice();

	VnVecDense(VnVec<float>* srcVec, cusparseHandle_t cusparseHandle);
	VnVecDense(const float initValue, int nVng, int* vngSizes, int fullLengthPadded, cusparseHandle_t cusparseHandle);
	~VnVecDense();
};


class Inference {
	cusparseHandle_t cusparseHandle = 0;
	cublasHandle_t cublasHandle = 0;
	
	Agds* agds;
	ConnSparse* connSparse;
	VnVecDense* prodVec;
	VnVecDense* revCountsVec;

	VecDense* AOn;
	VnVecDense* AVn;

	VecDense* counts;

	float* weightedAVnDev;

	void on2vn();
	void vn2vn();
	void vn2on();

public:
	void infere();
	void setupOnQuery(int* activatedOns, int nActivatedOns);

	Inference(Agds* agds);
	~Inference();
};