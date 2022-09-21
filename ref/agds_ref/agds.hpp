#pragma once

#include <vector>
#include "avbtree.hpp"


const float EPSILON = 0.0001;


struct VN {
	float value;
	float activation;
	float activation_tmp;

	VN() {}

	VN(float value) {
		this->value = value;
		this->activation = 0.0;
		this->activation_tmp = 0.0;
	}

	VN(const VN& other) {
		value = other.value;
		activation = other.activation;
		activation_tmp = other.activation_tmp;
	}

	VN& operator=(const VN& other) {
		if (this == &other) {
			return *this;
		}

		value = other.value;
		activation = other.activation;
		activation_tmp = other.activation_tmp;
		return *this;
	}

	bool operator<(const VN& other) {
		return value < other.value;
	}

	bool operator>(const VN& other) {
		return value > other.value;
	}

	float operator-(const VN& other) {
		return value - other.value;
	}

	bool operator==(const VN& other) {
		return std::fabs(value - other.value) < EPSILON;
	}
};


class VNG {
public:
	AVBTree<VN>::Iterator begin() { return tree.begin(); }
	AVBTree<VN>::Iterator end() { return tree.end(); }

	float range() {
		return tree.isEmpty() ? tree.max() - tree.min() : 0;
	}

	AVBValue<VN>* insert(float value) {
		return tree.insert(VN(value));
	}

	int size() {
		return tree.size;
	}

private:
	AVBTree<VN> tree;
};




class AGDS {
public:
	std::vector<VNG> vngs;
	std::vector<std::vector<AVBValue<VN>*>> CONN;

	int getNOn() {
		return n_on;
	}
	
	int getNVng() {
		return n_vng;
	}

	int getNVn() {
		int sum = 0;
		for (auto&& vng : vngs) {
			sum += vng.size();
		}
		return sum;
	}

	AGDS(float* data, int n_on, int n_vng)
	{
		this->n_on = n_on;
		this->n_vng = n_vng;

		vngs.resize(n_vng);
		CONN.reserve(n_on);

		for (int on_ix = 0; on_ix < n_on; on_ix++) {
			std::vector<AVBValue<VN>*> conn_column;
			conn_column.reserve(n_vng);
			CONN.push_back(conn_column);
		}

		for (int vng_ix = 0; vng_ix < n_vng; vng_ix++) {
			for (int on_ix = 0; on_ix < n_on; on_ix++) {
				AVBValue<VN>* vn = vngs[vng_ix].insert(data[n_on * vng_ix + on_ix]);
				CONN[on_ix].push_back(vn);
			}
		}
	}

private:
	int n_on;
	int n_vng;
};
