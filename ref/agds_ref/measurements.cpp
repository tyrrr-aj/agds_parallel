#include "measurements.hpp"


using std::chrono::high_resolution_clock;


int Measurer::startMeasurement() {
	int currentSize = startTimes.size();
	startTimes.resize(currentSize + 1);
	startTimes[currentSize] = high_resolution_clock::now();
	return currentSize;
}


void Measurer::endMeasurement(int measurementId) {
	high_resolution_clock::time_point endTime = high_resolution_clock::now();
	if (endTimes.size() <= measurementId) {
		endTimes.resize(measurementId + 1);
	}
	endTimes[measurementId] = endTime;
}


float Measurer::getElapsedTimeInSeconds(int measurementId) {
	return std::chrono::duration_cast<std::chrono::milliseconds>(endTimes[measurementId] - startTimes[measurementId]).count() / 1000.0;
}


float Measurer::getAvgTimeInSeconds() {
	float sum = 0.0;

	for (int measurementId = 0; measurementId < startTimes.size(); measurementId++) {
		sum += getElapsedTimeInSeconds(measurementId);
	}

	return sum / startTimes.size();
}
