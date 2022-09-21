#pragma once

#include "agds.hpp"


class Inference {
public:
	Inference(AGDS* agds);

	void setupOnQuery(int* activatedOns, int nActivatedOns);
	void infere();

	float* AOn;

	~Inference();

private:
	AGDS* agds;

	void on2vn();
	void vn2vn();
	void vn2on();
};