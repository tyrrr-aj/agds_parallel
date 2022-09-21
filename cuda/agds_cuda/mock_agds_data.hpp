#pragma once

// mock data

float* init_full_data(int n_groups, int n_on);
float* load_data(std::string path, int& nRows, int& nCols);


// mock queries

void mock_on_queries(int* activated_ons, int n_queries, int n_on, int n_activated_ons);
void mock_vn_queries_inactive_vngs(int* vn_queries_inactive_vngs, int n_queries, int n_groups, int n_activated_vngs);
void mock_vn_queries_vng(int* vn_queries_inactive_vngs, int* vn_queries_vng, int n_groups, int n_queries, int n_activated_vns_per_group);
