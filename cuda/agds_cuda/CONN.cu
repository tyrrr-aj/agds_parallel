#include "agdsgpu.cuh"

#include <cusparse_v2.h>
#include <cublas_v2.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>


void CONN::addON(int vnOffset) {
	cooNzValuesDev.append(1.0);
	cooRowIndicesDev.append(nOn);
	cooColIndicesDev.append(vnOffset);

	nOn++;
	if (vnOffset == nVn) {
		nVn++;
	}

	freeDescr();
}


cusparseSpMatDescr_t CONN::getDescr() {
	if (!isDescrReady) {
		CHECK_CUSPARSE(cusparseCreateCoo(
			&descr,
			nOn,
			nVn,
			nOn,
			cooRowIndicesDev.valuesDev,
			cooColIndicesDev.valuesDev,
			cooNzValuesDev.valuesDev,
			CUSPARSE_INDEX_32I,
			CUSPARSE_INDEX_BASE_ZERO,
			CUDA_R_32F
		));

		isDescrReady = true;
	}

	return descr;
}


void CONN::freeDescr() {
	if (isDescrReady) {
		CHECK_CUSPARSE(cusparseDestroySpMat(descr));
		isDescrReady = false;
	}
}


CONN::CONN() {
	CHECK_CUDA(cudaStreamCreate(&stream));
}


CONN::~CONN() {
	freeDescr();
	CHECK_CUDA(cudaStreamDestroy(stream));
}
