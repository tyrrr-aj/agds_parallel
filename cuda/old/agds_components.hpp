#pragma once

#include "utils.hpp"


class Conn {
	int** matrixCondensed;
	int** matrixExpanded;
	int nVng; // y size (condensed matrix)
	int nOn; // x size
	int nVn; // y size (expanded matrix)

	float* valuesCOO = NULL;
	int* rowIndicesCOO = NULL;
	int* colIndicesCOO = NULL;
	void fillCOO();

	void expandConn();

	public:
		int** getConnExpanded();
		int** getConnCondensed();

		int getXSize();
		int getYSizeCondensed();
		int getYSizeExpanded();

		float* getValuesCOO();
		int* getRowIndicesCOO();
		int* getColIndicesCOO();
		int getNNZ();

		static int** initMatrix(int nVng, int nOn);

		Conn(int nVng, int nOn, int nVn, int** connMatrixCondensed);
		~Conn();
};


template <typename T>
class VnVec {
protected:
	T* vector;
	int* vngStartingPoints;
	int* vngSizes;

public:
	int nVng; // public for use in VnVecDense

	int getSize() {
		return sum(vngSizes, nVng);
	}

	int getVngSize(int vngId) {
		return vngSizes[vngId];
	}

	int getVngOffset(int vngId) {
		return vngStartingPoints[vngId];
	}

	T* VnVec<T>::getFullVec() {
		return vector;
	}

	T* VnVec<T>::getVecForVng(int vngIdx) {
		return vector + vngStartingPoints[vngIdx];
	}

	static T** VnVec<T>::initVecPartsTmp(int nVng, int nOn) {
		T** vecPartsTmp = new T * [nVng];
		for (int vngIx = 0; vngIx < nVng; vngIx++) {
			vecPartsTmp[vngIx] = new T[nOn];
		}
		return vecPartsTmp;
	}

	VnVec(int nVng, int* vngSizes, T** vecPartsTmp) {
		this->vngSizes = vngSizes;
		this->nVng = nVng;

		vngStartingPoints = new int[nVng];
		cumulated_sum_shifted(vngSizes, nVng, vngStartingPoints);

		int nVn = sum(vngSizes, nVng);
		vector = new T[nVn];

		for (int vngIx = 0; vngIx < nVng; vngIx++) {
			for (int vnLocalIx = 0; vnLocalIx < vngSizes[vngIx]; vnLocalIx++) {
				vector[vngStartingPoints[vngIx] + vnLocalIx] = vecPartsTmp[vngIx][vnLocalIx];
			}
		}
	}

	VnVec(int nVng, int* vngSizes, const T initValue) {
		this->vngSizes = vngSizes;
		this->nVng = nVng;

		vngStartingPoints = new int[nVng];
		cumulated_sum_shifted(vngSizes, nVng, vngStartingPoints);

		int nVn = sum(vngSizes, nVng);
		vector = new T[nVn];

		for (int vnIx = 0; vnIx < nVn; vnIx++) {
			vector[vnIx] = initValue;
		}
	}

	~VnVec() {
		delete[] vector;
		delete[] vngStartingPoints;
	}
};


class ProductVec : public VnVec<float> {
	void computeProduct();
	void saveVngRanges(int nVng);

public:
	float* vngRanges;

	ProductVec(int nVng, int* vngSizes, float** valuesVecPartsTmp);
	~ProductVec();
};


class Agds {
	int nOn;
	int nVn;
	int nVng;

public:
	Conn* conn;
	ProductVec* productVec;
	VnVec<float>* countRevVec;
	int* vngSizes;

	int getNOn();
	int getNVn();
	int getNVng();

	Agds(int nVng, int nOn, float* data, float epsilon);
	~Agds();
};
