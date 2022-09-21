#include <iostream>
#include <cusparse_v2.h>
//#include <thrust/device_vector.h>
//#include <thrust/transform_reduce.h>
#include <cublas_v2.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "inference.cuh"


#define PRINT_AON(info) print_arr(AOn->values, info, AOn->length)
#define PRINT_AVN(info) print_arr(AVn->fullVec->values, info, AVn->fullVec->length)
#define PRINT_PHASE_SEP(next_phase) std::cout << std::endl << "Performing " << next_phase << std::endl << std::endl


__global__ void VecMulElementwise(float* a, float* b, float* res, int len) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len) {
		res[i] = a[i] * b[i];
	}
}

__constant__ const float alphabeta = 1.0;


Inference::Inference(Agds* const agds) {
	this->agds = agds;

	CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
	CHECK_CUBLAS(cublasCreate(&cublasHandle));

	connSparse = new ConnSparse(agds->conn, cusparseHandle);
	connSparse->setup();

	prodVec = new VnVecDense(agds->productVec, cusparseHandle);
	revCountsVec = new VnVecDense(agds->countRevVec, cusparseHandle);
	AVn = new VnVecDense(0.0, agds->getNVng(), agds->vngSizes, pad(agds->getNVn(), connSparse->myBlockDim), cusparseHandle);
	AOn = new VecDense(0.0, agds->getNOn(), pad(agds->getNOn(), connSparse->myBlockDim), cusparseHandle);
	counts = new VecDense(agds->countRevVec->getFullVec(), agds->getNVn(), cusparseHandle);

	// std::cout << "allocating: " << pad(AVn->fullVec->length, connSparse->myBlockDim) << " floats\n";
	CHECK_CUDA(cudaMalloc(&weightedAVnDev, pad(AVn->fullVec->length, connSparse->myBlockDim) * sizeof(float)));
}


Inference::~Inference() {
	delete connSparse;
	delete prodVec;
	delete revCountsVec;

	CHECK_CUDA(cudaFree(weightedAVnDev));
	CHECK_CUBLAS(cublasDestroy(cublasHandle));
	CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));
}


void Inference::setupOnQuery(int* activatedOns, int nActivatedOns) {
	for (int aOnIx = 0; aOnIx < nActivatedOns; aOnIx++) {
		AOn->setValue(activatedOns[aOnIx], 1.0);
	}
}


void Inference::infere() {
	/*print_arr(agds->conn->getConnExpanded(), "CONN", agds->conn->getYSizeExpanded(), agds->conn->getXSize());

	CHECK_CUDA(cudaMemcpy(AOn->values, AOn->valuesDev, AOn->length * sizeof(float), cudaMemcpyDeviceToHost));
	PRINT_AON("AOn before inference");
	PRINT_AVN("AVn before inference");

	PRINT_PHASE_SEP("on2vn");*/

	on2vn();

	/*CHECK_CUDA(cudaMemcpy(AVn->fullVec->values, AVn->fullVec->valuesDev, AVn->fullVec->length * sizeof(float), cudaMemcpyDeviceToHost));
	PRINT_AON("AOn after on2vn");
	PRINT_AVN("AVn after on2vn");

	PRINT_PHASE_SEP("vn2vn");*/

	vn2vn();

	/*CHECK_CUDA(cudaMemcpy(AVn->fullVec->values, AVn->fullVec->valuesDev, AVn->fullVec->length * sizeof(float), cudaMemcpyDeviceToHost));
	PRINT_AVN("AVn after vn2vn");

	PRINT_PHASE_SEP("vn2on");*/

	vn2on();

	/*CHECK_CUDA(cudaMemcpy(AOn->values, AOn->valuesDev, AOn->length * sizeof(float), cudaMemcpyDeviceToHost));
	PRINT_AON("AOn after inference");*/

	AVn->loadFromDevice();
	AOn->loadFromDevice();
}


void Inference::on2vn() {
	CHECK_CUSPARSE(cusparseSbsrmv(
		cusparseHandle,
		connSparse->matrixDir,
		CUSPARSE_OPERATION_NON_TRANSPOSE,
		connSparse->getNBlocksRows(),
		connSparse->getNBlocksCols(),
		connSparse->nnz,
		&alphabeta,
		connSparse->descrGen,
		connSparse->nzValuesBSRDev,
		connSparse->rowIndicesBSRDev,
		connSparse->colIndicesBSRDev,
		connSparse->myBlockDim,
		AOn->valuesDev,
		&alphabeta,
		AVn->fullVec->valuesDev
	));
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


// THRUST ATTEMPT (incomplete)
//float multiplyPair(thrust::tuple<float, float> weightAndActivation) {
//	return thrust::get<0>(weightAndActivation) * thrust::get<1>(weightAndActivation);
//}
//
//float weightFromProducts(thrust::tuple<float, float> products) {
//	float firstProd = thrust::get<0>(products);
//	float secondProd = thrust::get<1>(products);
//	return firstProd > secondProd ? secondProd / firstProd : firstProd / secondProd;
//}


void Inference::vn2vn() {
	for (int vngIdx = 0; vngIdx < agds->getNVng(); vngIdx++) {
		int vngSize = agds->vngSizes[vngIdx];

		/*std::cout << std::endl << "vn2vn for VNG " << vngIdx << " (length " << vngSize << "):" << std::endl;*/

		float* weightsMatDev;

	// 	std::cout << "allocating: " << vngSize * vngSize << " floats\n";
		CHECK_CUDA(cudaMalloc(&weightsMatDev, vngSize * vngSize * sizeof(float)));

		dim3 threadsPerBlock(16, 16);
		dim3 nBlocks((vngSize + threadsPerBlock.x - 1) / threadsPerBlock.x, (vngSize + threadsPerBlock.y - 1) / threadsPerBlock.y);
		computeWeights <<< nBlocks, threadsPerBlock >>> (prodVec->vecPartsVngs[vngIdx]->valuesDev, weightsMatDev, vngSize);

		/*print_arr(prodVec->vecPartsVngs[vngIdx]->values, "prodVec", vngSize);

		float* weightsMatHost = new float[vngSize * vngSize];
		CHECK_CUDA(cudaMemcpy(weightsMatHost, weightsMatDev, vngSize * vngSize * sizeof(float), cudaMemcpyDeviceToHost));
		print_arr(weightsMatHost, "weightsMat (column-major)", vngSize * vngSize);
		delete[] weightsMatHost;*/

		CHECK_CUBLAS(cublasSgemv(
			cublasHandle,
			CUBLAS_OP_N,
			vngSize,
			vngSize,
			&alphabeta,
			weightsMatDev,
			vngSize,
			AVn->vecPartsVngs[vngIdx]->valuesDev,
			1,
			&alphabeta,
			AVn->vecPartsVngs[vngIdx]->valuesDev,
			1
		));

		CHECK_CUDA(cudaFree(weightsMatDev));
		// std::cout << "freed: " << vngSize * vngSize << " floats\n";

		/* THRUST ATTEMPT (incomplete)
		auto productValuePairs = thrust::make_zip_iterator(sourceProducts, targetProducts);
		auto weights = thrust::make_transform_iterator(productValuePairs, multiplyPair);
		auto activationsAndWeights = thrust::make_zip_iterator(activations, weights);
		auto weightedActivations = thrust::make_transform_iterator(activationsAndWeights, multiplyPair);
		thrust::reduce_by_key(vnIds, vnIds + vngSize, weightedActivations, outKeys, newActivations);*/
	}
}


void Inference::vn2on() {
	const int blockSize = 512;
	int nBlocks = agds->getNVn() / blockSize + 1;
	VecMulElementwise <<< nBlocks, blockSize >>> (AVn->fullVec->valuesDev, counts->valuesDev, weightedAVnDev, agds->getNVn());

	CHECK_CUSPARSE(cusparseSbsrmv(
		cusparseHandle,
		connSparse->matrixDir,
		CUSPARSE_OPERATION_NON_TRANSPOSE,
		connSparse->getNBlocksCols(),
		connSparse->getNBlocksRows(),
		connSparse->nnz,
		&alphabeta,
		connSparse->descrGen,
		connSparse->nzValuesBSRTransposedDev,
		connSparse->rowIndicesBSRTransposedDev,
		connSparse->colIndicesBSRTransposedDev,
		connSparse->myBlockDim,
		weightedAVnDev,
		&alphabeta,
		AOn->valuesDev
	));
}
