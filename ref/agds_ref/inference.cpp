#include "inference.hpp"


Inference::Inference(AGDS* agds) {
	this->agds = agds;
	AOn = new float[agds->getNOn()];
}

Inference::~Inference() {
	delete AOn;
}

void Inference::setupOnQuery(int* activatedOns, int nActivatedOns) {
	for (int on_ix = 0; on_ix < agds->getNOn(); on_ix++) {
		AOn[on_ix] = 0.0;
	}

	for (auto&& vng : agds->vngs) {
		for (const auto& vn : vng) {
			vn->value.activation = 0.0;
			vn->value.activation_tmp = 0.0;
		}
	}

	for (int act_on_ix = 0; act_on_ix < nActivatedOns; act_on_ix++) {
		AOn[act_on_ix] = 1.0;
	}
}


void Inference::on2vn() {
	for (int vng_ix = 0; vng_ix < agds->getNVng(); vng_ix++) {
		for (int on_ix = 0; on_ix < agds->getNOn(); on_ix++) {
			agds->CONN[on_ix][vng_ix]->value.activation += AOn[on_ix];
		}
	}
}

void Inference::vn2vn() {
	for (auto&& vng : agds->vngs) {
		for (const auto& source_vn : vng) {
			AVBValue<VN>* curr = source_vn;
			AVBValue<VN>* neigh = source_vn->next;
			float propagatedActivation = source_vn->value.activation;
			float weight;

			// forward
			while (neigh != NULL) {
				weight = (vng.range() - (neigh->value.value - curr->value.value)) / vng.range();
				propagatedActivation *= weight;
				neigh->value.activation_tmp += propagatedActivation;

				curr = neigh;
				neigh = neigh->next;
			}

			// backward
			curr = source_vn;
			neigh = source_vn->prev;
			propagatedActivation = source_vn->value.activation;
			while (neigh != NULL) {
				weight = (vng.range() - (curr->value.value - neigh->value.value)) / vng.range();
				propagatedActivation *= weight;
				neigh->value.activation_tmp += propagatedActivation;

				curr = neigh;
				neigh = neigh->prev;
			}
		}

		for (const auto& vn : vng) {
			vn->value.activation += vn->value.activation_tmp;
		}
	}
}

void Inference::vn2on() {
	for (int vng_ix = 0; vng_ix < agds->getNVng(); vng_ix++) {
		for (int on_ix = 0; on_ix < agds->getNOn(); on_ix++) {
			AOn[on_ix] += agds->CONN[on_ix][vng_ix]->value.activation / (float) agds->CONN[on_ix][vng_ix]->count;
		}
	}
}


void Inference::infere() {
	on2vn();
	vn2vn();
	vn2on();
}