#pragma once

#include <string>
#include <vector>
#include <stdio.h>
#include <algorithm>


void cumulated_sum(int* arr, int len, int* res); // for [1,2,3,4,5] fills res with [1,3,6,10,15]
void cumulated_sum_shifted(int* arr, int len, int* res); // for [1,2,3,4,5] fills res with [0,1,3,6,10]
int sum(int* arr, int len);
int max_elem_index(int* arr, int len);
//std::vector<int> sort_indices(float* v, int n);

int n_elems_in_equal_split(int total_n_elements, int n_portions, int portion_index);

void print_arr(float* arr, std::string name, int len);
void print_arr(int* arr, std::string name, int len);
void print_arr(int** arr, std::string name, int nrows, int ncols);
void print_arr(float** arr, std::string name, int nrows, int ncols);

template<typename T>
int upper_bound(T* arr, int len, T value) {
    auto result_it = std::upper_bound(arr, arr + len, value);
    return std::distance(arr, result_it);
}

template<typename T>
std::vector<int> sort_indices(T* v, int n) {
    std::vector<int> idx(n);
    iota(idx.begin(), idx.end(), 0);

    stable_sort(idx.begin(), idx.end(),
        [v](int i1, int i2) {return v[i1] < v[i2]; });

    return idx;
}

int pad(int actualLength, int multipleOf);