#pragma once

#include <chrono>
#include <vector>


class Measurer {
	std::vector<std::chrono::high_resolution_clock::time_point> startTimes;
	std::vector<std::chrono::high_resolution_clock::time_point> endTimes;

public:
	int startMeasurement(); // returns measurement id
	void endMeasurement(int measurementId);

	float getElapsedTimeInSeconds(int measurementId);
	float getAvgTimeInSeconds();
};