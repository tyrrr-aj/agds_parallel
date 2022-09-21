#include "agdsgpu.cuh"

#include <cusparse_v2.h>
#include <cublas_v2.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>


VNG::VNG() {
	CHECK_CUDA(cudaStreamCreate(&stream));
	P = new VecDense<float>(stream);
	Indices = new VecDense<int>(stream);
	Nrev = new VecDense<float>(stream);
}

VNG::~VNG() {
	// freeAVnDescr();
	delete P;
	delete Indices;
	delete Nrev;
	CHECK_CUDA(cudaStreamDestroy(stream));
}


__global__ void updatePKernel(float* P, int* Indices, int newValueIdx, float newValueP, float pMod, int vecLen) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;

	if (x < vecLen) {
		int idx = Indices[x];

		if (newValueIdx >= idx) {
			Indices[x]++;
			P[x] *= pMod;
		}
	}
}


float weight(float leftValue, float rightValue, float valueRange) {
	return (valueRange - (rightValue - leftValue)) / valueRange;
}


void VNG::updateP(TreeInsertResult<float>& insertRes, float newValue) {
	AVBValue<float>* leftNeigh = insertRes.insertedValue->prev;
	AVBValue<float>* rightNeigh = insertRes.insertedValue->next;

	// float valueRange = tree.max() - tree.min();
	float valueRange = 1.0;
	float leftNeighValue = leftNeigh != NULL ? leftNeigh->value : insertRes.insertedValue->value;
	float rightNeighValue = rightNeigh != NULL ? rightNeigh->value : insertRes.insertedValue->value;

	float newValueP;

	if (leftNeigh == NULL) {
		newValueP = 1.0;
	}
	else {
		float leftNeighP;
		CHECK_CUDA(cudaMemcpyAsync(&leftNeighP, P->valuesDev + leftNeigh->offset, sizeof(float), cudaMemcpyDeviceToHost));
		newValueP = leftNeighP * weight(leftNeighValue, newValue, valueRange);
	}

	float pMod = weight(leftNeighValue, newValue, valueRange) * weight(newValue, rightNeighValue, valueRange) / weight(leftNeighValue, rightNeighValue, valueRange);

	if (rightNeigh != NULL) {
		const int threadsPerBlock = 256;
		int nBlocks = (P->len + threadsPerBlock - 1) / threadsPerBlock;

		updatePKernel<<<nBlocks, threadsPerBlock, 0, stream>>>(P->valuesDev, Indices->valuesDev, insertRes.index, newValueP, pMod, P->len);
	}

	P->append(newValueP);
	Indices->append(insertRes.index);
}


void VNG::updateNrev(TreeInsertResult<float>& insertRes) {
	float newNRevValue = 1.0 / insertRes.insertedValue->count;
	Nrev->set(insertRes.insertedValue->offset, newNRevValue);
}


void VNG::updateCONN(TreeInsertResult<float>& insertRes) {
	Conn.addON(insertRes.insertedValue->offset);
}


void VNG::addValue(float newValue) {
	TreeInsertResult<float> insertRes = tree.insert(newValue);

	if (insertRes.wasNewVnCreated) {
		updateP(insertRes, newValue);
	}
	updateNrev(insertRes);
	updateCONN(insertRes);

	freeAVnDescr();
}


int VNG::getNVn() {
	return tree.size();
}


cusparseDnVecDescr_t VNG::getAVnDescr() {
	if (!AVnDescrReady) {
		AVn = new float[getNVn()];
		CHECK_CUDA(cudaMallocAsync(&AVnDev, getNVn() * sizeof(float), 0));
		CHECK_CUDA(cudaMemsetAsync(AVnDev, 0, getNVn() * sizeof(float)));
		CHECK_CUSPARSE(cusparseCreateDnVec(&AVnDescr, getNVn(), AVnDev, CUDA_R_32F));
		AVnDescrReady = true;
	}

	return AVnDescr;
}


void VNG::resetAVn() {
	if (AVnDescrReady) {
		CHECK_CUDA(cudaMemsetAsync(AVnDev, 0, getNVn() * sizeof(float)));
	}
}


void VNG::loadAVnToHost() {
	CHECK_CUDA(cudaMemcpyAsync(AVn, AVnDev, getNVn() * sizeof(float), cudaMemcpyDeviceToHost));
}


void VNG::printAVn() {
	auto _ = getAVnDescr();

	CHECK_CUDA(cudaMemcpyAsync(AVn, AVnDev, getNVn() * sizeof(float), cudaMemcpyDeviceToHost));
	std::cout << "AVn: ";

	for (int i = 0; i < getNVn(); i++) {
		std::cout << AVn[i] << " ";
	}
	std::cout << std::endl;
}


void VNG::freeAVnDescr() {
	if (AVnDescrReady) {
		CHECK_CUSPARSE(cusparseDestroyDnVec(AVnDescr));
		CHECK_CUDA(cudaFreeAsync(AVnDev, 0));
		delete[] AVn;
		AVnDescrReady = false;
	}
}
