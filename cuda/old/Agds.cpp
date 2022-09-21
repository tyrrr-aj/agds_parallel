#include <vector>
#include <algorithm>
#include <numeric>
#include "utils.hpp"

#include "agds_components.hpp"


int build_tree(float* const dataColumn, const int nOn, const double epsilon, int nVnInPreviousVngs, int* const outputConnRow, float* const outputTree, int* const outputCounts);
float** convertToRevs(int** countsParts, int nVng, int* vngSizes);

Agds::Agds(int nVng, int nOn, float* data, float epsilon) {
    this->nOn = nOn;
    this->nVng = nVng;

    int** connMatrix = Conn::initMatrix(nVng, nOn);
    float** valuesVecPartsTmp = VnVec<float>::initVecPartsTmp(nVng, nOn);
    int** countsVecPartsTmp = VnVec<int>::initVecPartsTmp(nVng, nOn);
    vngSizes = new int[nVng];

    nVn = 0;
    for (int vngIx = 0; vngIx < nVng; vngIx++) {
        vngSizes[vngIx] = build_tree(data + nOn * vngIx, nOn, epsilon, nVn, connMatrix[vngIx], valuesVecPartsTmp[vngIx], countsVecPartsTmp[vngIx]);
        nVn += vngSizes[vngIx];
    }

    conn = new Conn(nVng, nOn, nVn, connMatrix);
    productVec = new ProductVec(nVng, vngSizes, valuesVecPartsTmp);

    float** revCountsVecPartsTmp = convertToRevs(countsVecPartsTmp, nVng, vngSizes);
    countRevVec = new VnVec<float>(nVng, vngSizes, revCountsVecPartsTmp);

    for (int vngIx = 0; vngIx < nVng; vngIx++) {
        delete[] valuesVecPartsTmp[vngIx];
        delete[] revCountsVecPartsTmp[vngIx];
    }
    delete[] valuesVecPartsTmp;
    delete[] revCountsVecPartsTmp;
}


Agds::~Agds() {
    delete conn;
    delete productVec;
    delete countRevVec;
    delete[] vngSizes;
}


int Agds::getNOn() {
    return nOn;
}


int Agds::getNVn() {
    return nVn;
}


int Agds::getNVng() {
    return nVng;
}


int build_tree(float* const dataColumn, const int nOn, const double epsilon, const int nVnInPreviousVngs, int* const outputConnRow, float* const outputTree, int* const outputCounts) {
    // mock implementation
    float* tree_tmp = new float[nOn];
    int* counts_tmp = new int[nOn];
    int distinct_count = 0;
    bool found;

    for (int on_ix = 0; on_ix < nOn; on_ix++) {
        found = false;

        for (int i = 0; i < distinct_count; i++) {
            if (fabs(tree_tmp[i] - dataColumn[on_ix]) < epsilon) {
                counts_tmp[i]++;
                outputConnRow[on_ix] = i;
                found = true;
                break;
            }
        }


        if (!found) {
            tree_tmp[distinct_count] = dataColumn[on_ix];
            counts_tmp[distinct_count] = 1;
            outputConnRow[on_ix] = distinct_count;
            distinct_count++;
        }
    }

    std::vector<int> indices = sort_indices(tree_tmp, distinct_count); // store ordered indices of VNs in ascending order, i.e. for tree [5., 1., 3.] it will be [1, 2, 0]

    int i_src = 0;
    for (auto i : indices) {
        outputTree[i_src] = tree_tmp[i];
        outputCounts[i_src] = counts_tmp[i];
        i_src++;
    }

    int* reverse_indices = new int[distinct_count]; // at position i it stores index to which VNi should go in sorted tree, i.e. for tree [5., 1., 3.] it will be [2, 0, 1]
    for (int vn_ix = 0; vn_ix < distinct_count; vn_ix++) {
        reverse_indices[indices[vn_ix]] = vn_ix;
    }

    for (int on_ix = 0; on_ix < nOn; on_ix++) {
        outputConnRow[on_ix] = reverse_indices[outputConnRow[on_ix]] + nVnInPreviousVngs;
    }

    delete[] tree_tmp;
    delete[] counts_tmp;
    delete[] reverse_indices;

    return distinct_count;
}


float** convertToRevs(int** countsParts, int nVng, int* vngSizes) {
    float** revVecPartsTmp = new float*[nVng];
    for (int vngIx = 0; vngIx < nVng; vngIx++) {
        revVecPartsTmp[vngIx] = new float[vngSizes[vngIx]];

        for (int vnIx = 0; vnIx < vngSizes[vngIx]; vnIx++) {
            revVecPartsTmp[vngIx][vnIx] = 1.0 / countsParts[vngIx][vnIx];
        }

        delete[] countsParts[vngIx];
    }

    delete[] countsParts;

    return revVecPartsTmp;
}