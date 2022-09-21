#include <stdexcept>
#include <vector>
#include <algorithm>
#include <numeric>

#include "agds_components.hpp"



int** Conn::getConnExpanded() {
	return matrixExpanded;
}


int** Conn::getConnCondensed() {
	return matrixCondensed;
}


int Conn::getXSize() {
	return nOn;
}


int Conn::getYSizeCondensed() {
	return nVng;
}


int Conn::getYSizeExpanded() {
	return nVn;
}


Conn::Conn(int nVng, int nOn, int nVn, int** connMatrixCondensed) {
	this->nVng = nVng;
	this->nOn = nOn;
	this->nVn = nVn;

	matrixCondensed = connMatrixCondensed;
	expandConn();
	fillCOO();
}


int** Conn::initMatrix(int nVng, int nOn) {
	int** matrixCondensedStatic = new int* [nVng];
	for (int i = 0; i < nVng; i++) {
		matrixCondensedStatic[i] = new int[nOn];
	}
	return matrixCondensedStatic;
}


void Conn::expandConn() {
	matrixExpanded = new int* [nVn];

	int connectedVnIx;

	for (int vnIx = 0; vnIx < nVn; vnIx++) {
		matrixExpanded[vnIx] = new int[nOn];

		for (int onIx = 0; onIx < nOn; onIx++) {
			matrixExpanded[vnIx][onIx] = 0;
		}
	}

	for (int vngIx = 0; vngIx < nVng; vngIx++) {
		for (int onIx = 0; onIx < nOn; onIx++) {
			connectedVnIx = matrixCondensed[vngIx][onIx];
			matrixExpanded[connectedVnIx][onIx] = 1;
		}
	}
}


Conn::~Conn() {
	for (int i = 0; i < nVng; i++) {
		delete[] matrixCondensed[i];
	}
	delete[] matrixCondensed;

	for (int i = 0; i < nVn; i++) {
		delete[] matrixExpanded[i];
	}
	delete[] matrixExpanded;

	delete[] valuesCOO;
	delete[] rowIndicesCOO;
	delete[] colIndicesCOO;
}


int Conn::getNNZ() {
	return nOn * nVng; // number of nonzero elements = number of VN<->ON connections = nOn * nVng
}


float* Conn::getValuesCOO() {
	return valuesCOO;
}


int* Conn::getRowIndicesCOO() {
	return rowIndicesCOO;
}


int* Conn::getColIndicesCOO() {
	return colIndicesCOO;
}


void Conn::fillCOO() {
	valuesCOO = new float[getNNZ()];
	rowIndicesCOO = new int[getNNZ()];
	colIndicesCOO = new int[getNNZ()];

	//int targetIndex;

	int* tempConn = new int[getNNZ()];
	int i = 0;

	for (int y = 0; y < getYSizeCondensed(); y++) {
		for (int x = 0; x < getXSize(); x++) {
			tempConn[i++] = matrixCondensed[y][x];

			/*targetIndex = y * getXSize() + x;

			valuesCOO[targetIndex] = 1.0;
			rowIndicesCOO[targetIndex] = matrixCondensed[y][x];
			colIndicesCOO[targetIndex] = x;*/
		}
	}

	std::vector<int> sortedIndices = sort_indices(tempConn, getNNZ());

	for (int connId = 0; connId < getNNZ(); connId++) {
		valuesCOO[connId] = 1.0;
		rowIndicesCOO[connId] = tempConn[sortedIndices[connId]];
		colIndicesCOO[connId] = sortedIndices[connId] % nOn;
	}

	delete[] tempConn;
}