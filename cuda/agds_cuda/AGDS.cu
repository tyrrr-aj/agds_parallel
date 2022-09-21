#include "agdsgpu.cuh"

#include <cusparse_v2.h>
#include <cublas_v2.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include "measurements.hpp"


__constant__ const float alphabeta = 1.0;


AGDS::AGDS(float* data, int n_on, int n_vng) {
	CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
	CHECK_CUBLAS(cublasCreate(&cublasHandle));

	this->n_vng = n_vng;
	vngs.resize(n_vng);

	float* observation = new float[n_vng];

	for (int on_ix = 0; on_ix < n_on; on_ix++) {
		for (int vng_ix = 0; vng_ix < n_vng; vng_ix++) {
			observation[vng_ix] = data[vng_ix * n_on + on_ix];
		}

		addObservation(observation);
	}

	delete[] observation;
}

AGDS::~AGDS() {
	// freeAOnDescr();

	CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
	CHECK_CUBLAS(cublasDestroy(cublasHandle));
}


void AGDS::addObservation(float* values) {
	// int mesId = measurer.startMeasurement();

	for (int vng_ix = 0; vng_ix < n_vng; vng_ix++) {
		vngs[vng_ix].addValue(values[vng_ix]);
	}

	n_on++;

	freeAOnDescr();

	// measurer.endMeasurement(mesId);

	/*if (n_on % 1000 == 0) {
		std::cout << n_on << ", " << measurer.getElapsedTimeInSeconds(mesId) << std::endl;
	}*/
}


void AGDS::setupOnQuery(int* activatedOns, int nActivatedOns) {
	if (!isAOnReady) {
		AOn = new float[n_vn()];
		CHECK_CUDA(cudaMalloc(&AOnDev, n_on * sizeof(float)));
		CHECK_CUSPARSE(cusparseCreateDnVec(&AOnDescr, n_on, AOnDev, CUDA_R_32F));
		isAOnReady = true;
	}

	const float initialActivation = 1.0;

	CHECK_CUDA(cudaMemset(AOnDev, 0, n_on * sizeof(float)));
	for (int act_on_ix = 0; act_on_ix < nActivatedOns; act_on_ix++) {
		CHECK_CUDA(cudaMemcpy(AOnDev + activatedOns[act_on_ix], &initialActivation, sizeof(float), cudaMemcpyHostToDevice));
	}

	for (auto&& vng : vngs) {
		vng.resetAVn();
	}
}


void AGDS::freeAOnDescr() {
	if (isAOnReady) {
		CHECK_CUSPARSE(cusparseDestroyDnVec(AOnDescr));
		delete[] AOn;
		CHECK_CUDA(cudaFree(AOnDev));
		isAOnReady = false;
	}
}


int AGDS::n_vn() {
	int n_vn = 0;
	for (auto&& vng : vngs) {
		n_vn += vng.getNVn();
	}
	return n_vn;
}


void AGDS::printAOn() {
	CHECK_CUDA(cudaMemcpy(AOn, AOnDev, n_on * sizeof(float), cudaMemcpyDeviceToHost));
	std::cout << "AOn: ";

	for (int i = 0; i < n_on; i++) {
		std::cout << AOn[i] << " ";
	}
	std::cout << std::endl;
}


void AGDS::loadResultsToHost() {
	for (auto&& vng : vngs) {
		vng.loadAVnToHost();
	}

	CHECK_CUDA(cudaMemcpy(AOn, AOnDev, n_on * sizeof(float), cudaMemcpyDeviceToHost));
}


void AGDS::infere() {
	/*for (auto& vng : vngs) {
		CHECK_CUDA(cudaEventCreate(&vng.vngReadyForVn2OnEvent));
	}*/
	
	on2vn();
	vn2vn();
	vn2on();

	loadResultsToHost();

	/*for (auto& vng : vngs) {
		CHECK_CUDA(cudaEventDestroy(vng.vngReadyForVn2OnEvent));
	}*/
}


void AGDS::on2vn() {
	cusparseOperation_t opTranspose = CUSPARSE_OPERATION_TRANSPOSE;
	cusparseSpMVAlg_t algorithm = CUSPARSE_SPMV_COO_ALG1;
	cudaDataType valueType = CUDA_R_32F;

	for (auto&& vng : vngs) {
		void* buffer;
		size_t bufferSize;

		// cudaStream_t on2vnStream = vng.stream;
		cudaStream_t on2vnStream = 0;

		// CHECK_CUSPARSE(cusparseSetStream(cusparseHandle, on2vnStream));
		CHECK_CUSPARSE(
			cusparseSpMV_bufferSize(
				cusparseHandle,
				opTranspose,
				&alphabeta,
				vng.Conn.getDescr(),
				AOnDescr,
				&alphabeta,
				vng.getAVnDescr(),
				valueType,
				algorithm,
				&bufferSize
			)
		);

		CHECK_CUDA(cudaMallocAsync(&buffer, bufferSize, on2vnStream));

		/*vng.Conn.cooNzValuesDev.print("COO vals");
		vng.Conn.cooRowIndicesDev.print("COO rows");
		vng.Conn.cooColIndicesDev.print("COO cols");

		printAOn();
		vng.printAVn();*/

		// CHECK_CUSPARSE(cusparseSetStream(cusparseHandle, on2vnStream));
		CHECK_CUSPARSE(cusparseSpMV(
			cusparseHandle,
			opTranspose,
			&alphabeta,
			vng.Conn.getDescr(),
			AOnDescr,
			&alphabeta,
			vng.getAVnDescr(),
			valueType,
			algorithm,
			buffer
		));

		CHECK_CUDA(cudaFreeAsync(buffer, on2vnStream));

		/*printAOn();
		vng.printAVn();*/
	}
}


__global__ void computeWeights(float* prodSingleVng, float* weights, int vngSize) {
	int targetVnIdx = blockIdx.x * blockDim.x + threadIdx.x;
	int sourceVnIdx = blockIdx.y * blockDim.y + threadIdx.y;

	if (targetVnIdx < vngSize && sourceVnIdx < vngSize) {
		weights[sourceVnIdx * vngSize + targetVnIdx] =
			targetVnIdx > sourceVnIdx ?
			prodSingleVng[targetVnIdx] / prodSingleVng[sourceVnIdx]
			: prodSingleVng[sourceVnIdx] / prodSingleVng[targetVnIdx];
	}
}

void AGDS::vn2vn() {
	for (auto&& vng : vngs) {
		int vngSize = vng.getNVn();

		// cudaStream_t vn2vnStream = vng.stream;
		cudaStream_t vn2vnStream = 0;

		float* weightsMatDev;
		CHECK_CUDA(cudaMallocAsync(&weightsMatDev, vngSize * vngSize * sizeof(float), vn2vnStream));

		dim3 threadsPerBlock(16, 16);
		dim3 nBlocks((vngSize + threadsPerBlock.x - 1) / threadsPerBlock.x, (vngSize + threadsPerBlock.y - 1) / threadsPerBlock.y);
		computeWeights <<<nBlocks, threadsPerBlock, 0, vn2vnStream >>> (vng.P->valuesDev, weightsMatDev, vngSize);

		// CHECK_CUBLAS(cublasSetStream(cublasHandle, vn2vnStream));
		CHECK_CUBLAS(cublasSgemv(
			cublasHandle,
			CUBLAS_OP_N,
			vngSize,
			vngSize,
			&alphabeta,
			weightsMatDev,
			vngSize,
			vng.AVnDev,
			1,
			&alphabeta,
			vng.AVnDev,
			1
		));

		// CHECK_CUDA(cudaEventRecord(vng.vngReadyForVn2OnEvent, vn2vnStream))
		CHECK_CUDA(cudaFreeAsync(weightsMatDev, vn2vnStream));
	}
}


__global__ void VecMulElementwise(float* a, float* b, float* res, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		res[i] = a[i] * b[i];
	}
}

void AGDS::vn2on() {
	cusparseOperation_t opTranspose = CUSPARSE_OPERATION_NON_TRANSPOSE;
	const float alphabeta = 1.0;
	cusparseSpMVAlg_t algorithm = CUSPARSE_SPMV_COO_ALG1;
	cudaDataType valueType = CUDA_R_32F;

	const int blockSize = 512;

	cudaStream_t mainStream = 0;
	// 	CHECK_CUDA(cudaStreamCreate(&mainStream));

	for (auto&& vng : vngs) {
		// cudaStream_t on2vnStream = vng.stream;
		cudaStream_t on2vnStream = 0;

		float* weightedAVnDev;
		CHECK_CUDA(cudaMallocAsync(&weightedAVnDev, vng.getNVn() * sizeof(float), on2vnStream));

		int nBlocks = vng.getNVn() / blockSize + 1;
		VecMulElementwise << < nBlocks, blockSize, 0, on2vnStream >> > (vng.AVnDev, vng.Nrev->valuesDev, weightedAVnDev, vng.getNVn());

		cusparseDnVecDescr_t weightedAVnDescr;
		// CHECK_CUSPARSE(cusparseSetStream(cusparseHandle, on2vnStream));
		CHECK_CUSPARSE(cusparseCreateDnVec(&weightedAVnDescr, vng.getNVn(), weightedAVnDev, CUDA_R_32F));

		void* buffer;
		size_t bufferSize;

		// CHECK_CUSPARSE(cusparseSetStream(cusparseHandle, on2vnStream));
		CHECK_CUSPARSE(
			cusparseSpMV_bufferSize(
				cusparseHandle,
				opTranspose,
				&alphabeta,
				vng.Conn.getDescr(),
				weightedAVnDescr,
				&alphabeta,
				AOnDescr,
				valueType,
				algorithm,
				&bufferSize
			)
		);

		CHECK_CUDA(cudaMallocAsync(&buffer, bufferSize, on2vnStream));

		// CHECK_CUDA(cudaStreamWaitEvent(mainStream, vng.vngReadyForVn2OnEvent));
		// CHECK_CUSPARSE(cusparseSetStream(cusparseHandle, mainStream));
		CHECK_CUSPARSE(cusparseSpMV(
			cusparseHandle,
			opTranspose,
			&alphabeta,
			vng.Conn.getDescr(),
			weightedAVnDescr,
			&alphabeta,
			AOnDescr,
			valueType,
			algorithm,
			buffer
		));

		CHECK_CUDA(cudaFreeAsync(buffer, mainStream));
		
		// CHECK_CUSPARSE(cusparseSetStream(cusparseHandle, mainStream));
		CHECK_CUSPARSE(cusparseDestroyDnVec(weightedAVnDescr));
		CHECK_CUDA(cudaFreeAsync(weightedAVnDev, mainStream));
	}

	// CHECK_CUDA(cudaStreamDestroy(mainStream));
}
