#include <cuda_runtime.h>
#include <cusparse_v2.h>


#include "inference.cuh"


void ConnSparse::setup() {
	createCOO();
	coo2csr();
	transposeCSR();

	/*float csrNzValsHost[40], csrNzValsTransHost[40];
	int csrRowHost[23], csrRowTransHost[11];
	int csrColHost[40], csrColTransHost[40];*/
	
	/*CHECK_CUDA(cudaMemcpy(csrNzValsHost, nzValuesCOOCSRDev, 40 * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(csrRowHost, rowIndicesCSRDev, 23 * sizeof(int), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(csrColHost, colIndicesCOOCSRDev, 40 * sizeof(int), cudaMemcpyDeviceToHost));

	CHECK_CUDA(cudaMemcpy(csrNzValsTransHost, nzValuesCSRTransposedDev, 40 * sizeof(float), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(csrRowTransHost, rowIndicesCSRTransposedDev, 11 * sizeof(int), cudaMemcpyDeviceToHost));
	CHECK_CUDA(cudaMemcpy(csrColTransHost, colIndicesCSRTransposedDev, 40 * sizeof(int), cudaMemcpyDeviceToHost));*/

	// print_arr(csrNzValsHost, "nz", 40);
	//print_arr(csrRowHost, "row", 23);
	//print_arr(csrColHost, "col", 40);

	//// print_arr(csrNzValsTransHost, "nz T", 40);
	//print_arr(csrRowTransHost, "row T", 11);
	//print_arr(csrColTransHost, "col T", 40);

	csr2bsr( // original matrix
		nRows,
		nCols,
		nzValuesCOOCSRDev,
		rowIndicesCSRDev,
		colIndicesCOOCSRDev,
		nzValuesBSRDev,
		rowIndicesBSRDev,
		colIndicesBSRDev
	);

	csr2bsr( // transposed matrix
		nCols,
		nRows,
		nzValuesCSRTransposedDev,
		rowIndicesCSRTransposedDev,
		colIndicesCSRTransposedDev,
		nzValuesBSRTransposedDev,
		rowIndicesBSRTransposedDev,
		colIndicesBSRTransposedDev
	);
}


void ConnSparse::createCOO() {
	CHECK_CUSPARSE(cusparseCreateMatDescr(&descrGen));
	CHECK_CUSPARSE(cusparseSetMatType(descrGen, CUSPARSE_MATRIX_TYPE_GENERAL));
	CHECK_CUSPARSE(cusparseSetMatIndexBase(descrGen, CUSPARSE_INDEX_BASE_ZERO));

	CHECK_CUDA(cudaMalloc(&nzValuesCOOCSRDev, nnz * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&rowIndicesCOODev, nnz * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&colIndicesCOOCSRDev, nnz * sizeof(int)));

	CHECK_CUDA(cudaMemcpy(nzValuesCOOCSRDev, nzValuesCOO, nnz * sizeof(float), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(rowIndicesCOODev, rowIndicesCOO, nnz * sizeof(int), cudaMemcpyHostToDevice));
	CHECK_CUDA(cudaMemcpy(colIndicesCOOCSRDev, colIndicesCOO, nnz * sizeof(int), cudaMemcpyHostToDevice));

	CHECK_CUSPARSE(cusparseCreateCoo(
		&descrSparse,
		nRows,
		nCols,
		nnz,
		rowIndicesCOODev,
		colIndicesCOOCSRDev,
		nzValuesCOOCSRDev,
		CUSPARSE_INDEX_32I,
		CUSPARSE_INDEX_BASE_ZERO,
		CUDA_R_32F
	));
}


void ConnSparse::coo2csr() {
	CHECK_CUDA(cudaMalloc(&rowIndicesCSRDev, (nRows + 1) * sizeof(int)));

	CHECK_CUSPARSE(cusparseXcoo2csr(
		cusparseHandle,
		rowIndicesCOODev,
		nnz,
		nRows,
		rowIndicesCSRDev,
		CUSPARSE_INDEX_BASE_ZERO
	));

	CHECK_CUDA(cudaFree(rowIndicesCOODev));
}


void ConnSparse::transposeCSR() {
	CHECK_CUDA(cudaMalloc(&nzValuesCSRTransposedDev, nnz * sizeof(float)));
	CHECK_CUDA(cudaMalloc(&colIndicesCSRTransposedDev, nnz * sizeof(int)));
	CHECK_CUDA(cudaMalloc(&rowIndicesCSRTransposedDev, (nCols + 1) * sizeof(int)));

	CHECK_CUDA(cudaMemset(colIndicesCSRTransposedDev, 0, nnz * sizeof(int)));
	CHECK_CUDA(cudaMemset(rowIndicesCSRTransposedDev, 0, (nCols + 1) * sizeof(int)));

	cusparseCsr2CscAlg_t const alg = CUSPARSE_CSR2CSC_ALG1;
	cusparseAction_t const action = CUSPARSE_ACTION_NUMERIC;
	size_t bufferSize;

	CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(
		cusparseHandle,
		nRows,
		nCols,
		nnz,
		nzValuesCOOCSRDev,
		rowIndicesCSRDev,
		colIndicesCOOCSRDev,
		nzValuesCSRTransposedDev,
		rowIndicesCSRTransposedDev,
		colIndicesCSRTransposedDev,
		CUDA_R_32F,
		action,
		CUSPARSE_INDEX_BASE_ZERO,
		alg,
		&bufferSize
	));

	float* buffer;
	CHECK_CUDA(cudaMalloc(&buffer, bufferSize));

	CHECK_CUSPARSE(cusparseCsr2cscEx2(
		cusparseHandle,
		nRows,
		nCols,
		nnz,
		nzValuesCOOCSRDev,
		rowIndicesCSRDev,
		colIndicesCOOCSRDev,
		nzValuesCSRTransposedDev,
		rowIndicesCSRTransposedDev,
		colIndicesCSRTransposedDev,
		CUDA_R_32F,
		action,
		CUSPARSE_INDEX_BASE_ZERO,
		alg,
		buffer
	));

	CHECK_CUDA(cudaFree(buffer));
}


void ConnSparse::csr2bsr(
	int nRows,
	int nCols,
	float* nzValuesCSRDev,
	int* rowIndicesCSRDev,
	int* colIndicesCSRDev,
	float*& nzValuesBSRDev,
	int*& rowIndicesBSRDev,
	int*& colIndicesBSRDev
) {
	int base, nnzb;
	int vBlockNum = (nRows + myBlockDim - 1) / myBlockDim;

	// std::cout << "allocating: " << vBlockNum + 1 << " ints\n";
	CHECK_CUDA(cudaMalloc(&rowIndicesBSRDev, (vBlockNum + 1) * sizeof(int)));

	// copied from cuSPARSE doc
	int* nnzTotalDevHostPtr = &nnzb;

	CHECK_CUSPARSE(cusparseXcsr2bsrNnz(
		cusparseHandle,
		matrixDir,
		nRows,
		nCols,
		descrGen,
		rowIndicesCSRDev,
		colIndicesCSRDev,
		myBlockDim,
		descrGen,
		rowIndicesBSRDev,
		// &nnzb // possibly should be guraded with alternative read mode (see doc for cusparse<t>csr2bsr() )
		nnzTotalDevHostPtr
	));

	// copied from cuSPARSE doc
	if (NULL != nnzTotalDevHostPtr) {
		nnzb = *nnzTotalDevHostPtr;
	}
	else {
		cudaMemcpy(&nnzb, rowIndicesBSRDev + vBlockNum, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(&base, rowIndicesBSRDev, sizeof(int), cudaMemcpyDeviceToHost);
		nnzb -= base;
	}

	// std::cout << "allocating: " << nnzb << " ints\n";
	CHECK_CUDA(cudaMalloc(&colIndicesBSRDev, nnzb * sizeof(int)));
	// std::cout << "allocating: " << myBlockDim * myBlockDim * nnzb << " floats\n";
	CHECK_CUDA(cudaMalloc(&nzValuesBSRDev, (myBlockDim * myBlockDim * nnzb) * sizeof(float)));

	CHECK_CUSPARSE(cusparseScsr2bsr(
		cusparseHandle,
		matrixDir,
		nRows,
		nCols,
		descrGen,
		nzValuesCSRDev,
		rowIndicesCSRDev,
		colIndicesCSRDev,
		myBlockDim,
		descrGen,
		nzValuesBSRDev,
		rowIndicesBSRDev,
		colIndicesBSRDev
	));

	CHECK_CUDA(cudaFree(nzValuesCSRDev));
	CHECK_CUDA(cudaFree(rowIndicesCSRDev));
	CHECK_CUDA(cudaFree(colIndicesCSRDev));
}


ConnSparse::ConnSparse(Conn* const conn, cusparseHandle_t cusparseHandle) {
	this->cusparseHandle = cusparseHandle;
	nRows = conn->getYSizeExpanded();
	nCols = conn->getXSize();

	nnz = conn->getNNZ();
	nzValuesCOO = conn->getValuesCOO();
	rowIndicesCOO = conn->getRowIndicesCOO();
	colIndicesCOO = conn->getColIndicesCOO();
}


ConnSparse::~ConnSparse() {
	CHECK_CUDA(cudaFree(nzValuesBSRDev));
	CHECK_CUDA(cudaFree(rowIndicesBSRDev));
	CHECK_CUDA(cudaFree(colIndicesBSRDev));

	CHECK_CUDA(cudaFree(nzValuesBSRTransposedDev));
	CHECK_CUDA(cudaFree(rowIndicesBSRTransposedDev));
	CHECK_CUDA(cudaFree(colIndicesBSRTransposedDev));

	CHECK_CUSPARSE(cusparseDestroyMatDescr(descrGen));
}


int ConnSparse::getNBlocksRows() {
	return (nRows + myBlockDim - 1) / myBlockDim;
}

int ConnSparse::getNBlocksCols() {
	return (nCols + myBlockDim - 1) / myBlockDim;
}
