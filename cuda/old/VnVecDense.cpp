#include <cuda_runtime.h>
#include <cusparse_v2.h>

#include "inference.cuh"


VnVecDense::VnVecDense(VnVec<float>* srcVec, cusparseHandle_t cusparseHandle) {
	setup(srcVec, srcVec->getSize(), cusparseHandle);
}


VnVecDense::VnVecDense(const float initValue, int nVng, int* vngSizes, int fullLengthPadded, cusparseHandle_t cusparseHandle) {
	VnVec<float>* adHocVnVec = new VnVec<float>(nVng, vngSizes, initValue);
	isSrcVecCreatedInternally = true;

	setup(adHocVnVec, fullLengthPadded, cusparseHandle);
}


VnVecDense::~VnVecDense() {
	delete fullVec;

	for (int vngIx = 0; vngIx < srcVec->nVng; vngIx++) {
		delete vecPartsVngs[vngIx];
	}

	delete[] vecPartsVngs;

	if (isSrcVecCreatedInternally) {
		delete srcVec;
	}
}


void VnVecDense::setup(VnVec<float>* srcVec, int fullLengthPadded, cusparseHandle_t cusparseHandle) {
	this->cusparseHandle = cusparseHandle;
	this->srcVec = srcVec;

	fullVec = new VecDense(srcVec->getFullVec(), srcVec->getSize(), fullLengthPadded, cusparseHandle);

	vecPartsVngs = new VecDense*[srcVec->nVng];
	for (int vngIx = 0; vngIx < srcVec->nVng; vngIx++) {
		vecPartsVngs[vngIx] = new VecDense(srcVec->getVecForVng(vngIx), fullVec->valuesDev + srcVec->getVngOffset(vngIx), srcVec->getVngSize(vngIx), cusparseHandle);
	}
}


void VnVecDense::loadFromDevice() {
	fullVec->loadFromDevice();
}

void VnVecDense::loadToDevice() {
	fullVec->loadToDevice();
}
