#include "agds_components.hpp"


ProductVec::ProductVec(int nVng, int* vngSizes, float** valuesVecPartsTmp)
	: VnVec<float>(nVng, vngSizes, valuesVecPartsTmp)
{
	saveVngRanges(nVng);
	computeProduct();
}


void ProductVec::saveVngRanges(int nVng) {
	vngRanges = new float[nVng];

	float vngMinimum, vngMaximum;

	for (int vngIx = 0; vngIx < nVng; vngIx++) {
		vngMinimum = vector[vngStartingPoints[vngIx]];
		vngMaximum = vector[vngStartingPoints[vngIx] + vngSizes[vngIx] - 1];
		vngRanges[vngIx] = vngMaximum - vngMinimum;
	}
}


void ProductVec::computeProduct() {
	float lastValue, productValue;
	int vnGlobalIx;

	for (int vngIx = 0; vngIx < nVng; vngIx++) {
		lastValue = vector[vngStartingPoints[vngIx]];
		vector[vngStartingPoints[vngIx]] = (float)1.0;

		for (int vnLocalIx = 1; vnLocalIx < vngSizes[vngIx]; vnLocalIx++) {
			vnGlobalIx = vngStartingPoints[vngIx] + vnLocalIx;

			productValue = (vngRanges[vngIx] - (vector[vnGlobalIx] - lastValue)) / vngRanges[vngIx] * vector[vnGlobalIx - 1];
			lastValue = vector[vnGlobalIx];
			vector[vnGlobalIx] = productValue;
		}
	}
}


ProductVec::~ProductVec() {
	delete[] vngRanges;
}