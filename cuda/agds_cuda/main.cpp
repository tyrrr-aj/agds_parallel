#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

#include "mock_agds_data.hpp"
// #include "agds_components.hpp"
// #include "inference.cuh"
#include "agdsgpu.cuh"
#include "measurements.hpp"

const int SEED = 42;

const int K = 1;
const int N_QUERIES = 10;
int N_ACTIVATED_VNS_PER_GROUP = 2;
int N_ACTIVATED_ONS = 5;

float EPSILON;


int main(int argc, char** argv) {
	float* data;

	int N_ON, N_VNG;

	if (argc > 1) {
		data = load_data(argv[1], N_ON, N_VNG);

		EPSILON = argc > 2 ? std::stof(argv[2]) : 0.0001;
	}
	else {
		N_ON = 500;
		N_VNG = 4;
		EPSILON = (float)0.0001;

		data = init_full_data(N_VNG, N_ON);
	}

	int N_ACTIVATED_VNGS = N_VNG - 1;

	Measurer measurer;
	int constructionMesId, inferenceMesId;

	/*Agds agds = Agds(N_VNG, N_ON, data, EPSILON);
	Inference inference(&agds);*/

	constructionMesId = measurer.startMeasurement();
	AGDS agds(data, N_ON, N_VNG);
	measurer.endMeasurement(constructionMesId);

	std::cout
		<< std::fixed
		<< std::setprecision(2)
		<< "n_on: "
		<< N_ON
		<< ", n_vng: "
		<< N_VNG
		<< ", n_vn: "
		<< agds.n_vn()
		<< "\tconstruction time: "
		<< measurer.getElapsedTimeInSeconds(constructionMesId)
		<< "s"
		<< std::endl;

	/*print_arr(agds.conn->getConnExpanded(), "conn", agds.getNVn(), agds.getNOn());*/

	int* query_activated_ons = new int[N_ACTIVATED_ONS * N_QUERIES];

	for (int qIx = 0; qIx < N_QUERIES; qIx++) {
		mock_on_queries(query_activated_ons, N_QUERIES, N_ON, N_ACTIVATED_ONS);

		inferenceMesId = measurer.startMeasurement();
		// inference.setupOnQuery(query_activated_ons + N_ACTIVATED_ONS * qIx, N_ACTIVATED_ONS);
		// inference.infere();
		agds.setupOnQuery(query_activated_ons + N_ACTIVATED_ONS * qIx, N_ACTIVATED_ONS);
		agds.infere();
		measurer.endMeasurement(inferenceMesId);

		if (argc == 1) {
			std::cout
				<< std::fixed
				<< std::setprecision(2)
				<< "Measurement "
				<< inferenceMesId
				<< " - elapsed time: "
				<< measurer.getElapsedTimeInSeconds(inferenceMesId)
				<< "s"
				<< std::endl;
		}
	}

	if (argc == 1) {
		std::cout
			<< std::fixed
			<< std::setprecision(2)
			<< std::endl
			<< "Average inference time: "
			<< measurer.getAvgTimeInSeconds()
			<< "s"
			<< std::endl;
	}
	else {
		std::cout
			<< std::fixed
			<< std::setprecision(2)
			<< "n_on: "
			<< N_ON
			<< ", n_vng: "
			<< N_VNG
			<< ", n_vn: "
			<< agds.n_vn()
			<< "\tavg inference time: "
			<< measurer.getAvgTimeInSeconds()
			<< "s"
			<< std::endl;
	}

	delete[] query_activated_ons;

	return 0;
}