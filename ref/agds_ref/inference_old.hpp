#pragma once

#include <string>

#include "agds_components.hpp"


class InferenceOld {
	Agds* agds;

	float* AVnTemp;

	int vnGlobalIx(int vnLocalIx, int vngIx);

	void on2vn();
	void vn2vn();
	void vn2on();

public:
	void infere();
	void setupOnQuery(int* activatedOns, int nActivatedOns);

	float* AOn;
	float* AVn;

	InferenceOld(Agds* agds);
	~InferenceOld();
};
