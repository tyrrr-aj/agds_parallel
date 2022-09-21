#include "inference_old.hpp"

void InferenceOld::on2vn() {
	int targetVn;

	for (int onIx = 0; onIx < agds->getNOn(); onIx++) {
		for (int vngIx = 0; vngIx < agds->getNVng(); vngIx++) {
			targetVn = agds->conn->getConnCondensed()[vngIx][onIx];
			AVn[targetVn] += AOn[onIx];
		}
	}
}

int InferenceOld::vnGlobalIx(int vnLocalIx, int vngIx) {
	return agds->vngOffsets[vngIx] + vnLocalIx;
}

void InferenceOld::vn2vn() {
	int sourceVnGlobalIx, targetVnGlobalIx;
	float weight;

	for (int vnIx = 0; vnIx < agds->getNVn(); vnIx++) {
		AVnTemp[vnIx] = 0.0;
	}

	for (int vngIx = 0; vngIx < agds->getNVng(); vngIx++) {
		for (int sourceVnLocalIx = 0; sourceVnLocalIx < agds->vngSizes[vngIx]; sourceVnLocalIx++) {
			for (int targetVnLocalIx = sourceVnLocalIx + 1; targetVnLocalIx < agds->vngSizes[vngIx]; targetVnLocalIx++) {
				sourceVnGlobalIx = vnGlobalIx(sourceVnLocalIx, vngIx);
				targetVnGlobalIx = vnGlobalIx(targetVnLocalIx, vngIx);

				weight = agds->productVec->getVecForVng(vngIx)[targetVnLocalIx] / agds->productVec->getVecForVng(vngIx)[sourceVnLocalIx];
				
				AVnTemp[targetVnGlobalIx] += AVn[sourceVnGlobalIx] * weight;
				AVnTemp[sourceVnGlobalIx] += AVn[targetVnGlobalIx] * weight;
			}
		}
	}

	for (int vnIx = 0; vnIx < agds->getNVn(); vnIx++) {
		AVn[vnIx] += AVnTemp[vnIx];
	}
}

void InferenceOld::vn2on() {
	int sourceVnIx;

	for (int onIx = 0; onIx < agds->getNOn(); onIx++) {
		for (int vngIx = 0; vngIx < agds->getNVng(); vngIx++) {
			sourceVnIx = agds->conn->getConnCondensed()[vngIx][onIx];
			AOn[onIx] += AVn[sourceVnIx] * agds->countRevVec->getFullVec()[sourceVnIx];
		}
	}
}

void InferenceOld::infere() {
	on2vn();
	vn2vn();
	vn2on();
}

void InferenceOld::setupOnQuery(int* activatedOns, int nActivatedOns) {
	for (int onIx = 0; onIx < agds->getNOn(); onIx++) {
		AOn[onIx] = 0.0;
	}

	for (int vnIx = 0; vnIx < agds->getNVn(); vnIx++) {
		AVn[vnIx] = 0.0;
	}

	for (int actOnIx = 0; actOnIx < nActivatedOns; actOnIx++) {
		AOn[actOnIx] = 1.0;
	}
}


InferenceOld::InferenceOld(Agds* agds) {
	this->agds = agds;
	AVn = new float[agds->getNVn()];
	AOn = new float[agds->getNOn()];

	AVnTemp = new float[agds->getNVn()];
}

InferenceOld::~InferenceOld() {
	delete[] AVn;
	delete[] AOn;

	delete[] AVnTemp;
}
