#include <mpi.h>
#include <mpe.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <random>
#include <vector>
#include <fstream>
#include <iomanip>

#include "scan_prod_mpi.hpp"
#include "inference.hpp"
#include "utils.hpp"
#include "logging_states.hpp"
#include "mock_agds_data.hpp"

int N_ON, N_VNG;

// const int N_QUERIES = 2;
const int N_QUERIES = 10;
const int ACTIVATED_VNS_PER_GROUP = 4;
int ACTIVATED_VNGS;
const int ACTIVATED_ONS = 5;
const int K = 1;

const int SEED = 42;

const int MASTERS_TAG = 0;
const int AGDS_MASTER_RANK = 0;

double EPSILON;


#pragma region Debug

void debug_printf_from_vng_leaders(const int rank, const int vng_rank, const int vng_id, const int n_vn_vng) {
    if (vng_rank == 0) {
        printf("I'm %d, 0 in group %d, and my VNG consists of %d VNs\n", rank, vng_id, n_vn_vng);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void debug_printf_from_all_workers(const int rank, const int vng_id, const int n_vn_p) {
    printf("I'm %d, working for %d, and I've received %d values\n", rank, vng_id, n_vn_p);
}

#pragma endregion


#pragma region Logging

void init_logging() {
    MPE_Init_log();
    init_events();
}

void save_log(int size) {
    const int log_file_max_length = 50;
    char log_file_name[log_file_max_length];
    snprintf(log_file_name, log_file_max_length, "inference%d", size);

    MPE_Finish_log(log_file_name);
}

#pragma endregion


#pragma region Measurements

double make_measurement() {
    MPI_Barrier(MPI_COMM_WORLD);
    return MPI_Wtime();
}

void report_times(int rank, double start_time, double end_time) {
    if (rank == AGDS_MASTER_RANK) {
        printf("Avg time: %.2fs\n", (end_time - start_time) / N_QUERIES);
    }
}

void report_times_expanded(const char* stage, int rank, int n_vn, int size, double start_time, double end_time) {
    if (rank == AGDS_MASTER_RANK) {
        std::ofstream output;
        output.open("../../report.txt", std::ios::out | std::ios::app);
        
        output << std::fixed;
		output << std::setprecision(2);
        output << "n_on: " << N_ON;
        output << ", n_vng: " << N_VNG; 
        output << ", n_vn: " << n_vn;
        output << ", n_proc: " << size;
        output << "\tavg " << stage << " time: " << (end_time - start_time) / N_QUERIES;
        output << std::endl;

        output.close();
    }
}



#pragma endregion


#pragma region Generic functions

void mpi_size_and_rank(const MPI_Comm &comm, int &size, int &rank) {
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
}

#pragma endregion


#pragma region Individual computations

void global_vn_indices_vn_to_vn_and_vn_to_on(const int vng_id, const int vng_rank, const int vng_size, const int n_vn_p, int* const Vng_n_vns, int* const Vns_vn_to_vn_and_vn_to_on_proc) {
    int n_vn_vng = Vng_n_vns[vng_id];
    int vng_offset = sum(Vng_n_vns, vng_id); // total number of VNs in preceeding VNGs

    for (int vn_loc_ix = 0; vn_loc_ix < n_vn_p; vn_loc_ix++) {
        Vns_vn_to_vn_and_vn_to_on_proc[vn_loc_ix] = vng_offset + vng_size * vn_loc_ix + vng_rank;
    }
}


#pragma endregion


#pragma region VNG group

void split_processes_into_vngs(int* const worker_spread, const int rank, int &vng_id, int &vng_size, int &vng_rank, MPI_Comm &vng_comm) {
    MPI_Scatter(worker_spread, 1, MPI_INT, &vng_id, 1, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);
    
    MPI_Comm_split(MPI_COMM_WORLD, vng_id, rank, &vng_comm);
    MPI_Comm_size(vng_comm, &vng_size);
    MPI_Comm_rank(vng_comm, &vng_rank);
}

void broadcast_number_of_vns_in_own_vng(int& n_vn_vng, const MPI_Comm &vng_comm) {
    // Each VNG leader informs the rest of processes in VNG about the number of VNs in this VNG
    MPI_Bcast(&n_vn_vng, 1, MPI_INT, 0, vng_comm);
}

void compute_vn_prod(double* const tree, const int n_vn_vng, double* &Vn_prod_from_scan_p, double &vn_range, int &n_vn_p, const MPI_Comm &vng_comm) {
    // compute VN_prod_from_scan_p vector via scan - it's sorted, which is not the desired order
    scan_prod_mpi(tree, n_vn_vng, Vn_prod_from_scan_p, vn_range, n_vn_p, 0, vng_comm);
}

void broadcast_number_of_vns_in_all_vngs(int* const Vng_n_vns) {
    // Vector with numbers of VNs in each VNG is shared with all processes (by master)
    MPI_Bcast(Vng_n_vns, N_VNG, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);
}

void gather_number_of_vns_per_proc(const int n_vn_p, int* const N_vn_of_ps_in_vng, int* &displ_vng_p, int vng_rank, int vng_size, const MPI_Comm &vng_comm) {
    //      gathering information about number of VNs in each process within VNG
    MPI_Gather(&n_vn_p, 1, MPI_INT, N_vn_of_ps_in_vng, 1, MPI_INT, 0, vng_comm);

    if (vng_rank == 0) {
        displ_vng_p = new int[vng_size];
        cumulated_sum_shifted(N_vn_of_ps_in_vng, vng_size, displ_vng_p);
    }
}

void gather_full_vn_prod_from_scan_vng(double* const Vn_prod_from_scan_p, const int n_vn_p, double* const Vn_prod_from_scan_vng,
        int* const N_vn_of_ps_in_vng, int* const displ_vng_p, const MPI_Comm &vng_comm) {
    //      gathering global Vn_prod_from_scan for each VNG
    MPI_Gatherv(Vn_prod_from_scan_p, n_vn_p, MPI_DOUBLE, Vn_prod_from_scan_vng, N_vn_of_ps_in_vng, displ_vng_p, MPI_DOUBLE, 0, vng_comm);
}

void reorder_vn_prod_vng(const int vng_size, const int vng_rank, const int n_vn_vng, double* const Vn_prod_vng, 
        double* const Vn_prod_from_scan_vng, int* const N_vn_vng, int* &N_vn_vng_reordered, int* const Vns_distribution_vng, int* &Vns_distribution_vng_reordered) {
    if (vng_rank == 0) {
        N_vn_vng_reordered = new int[n_vn_vng];
        Vns_distribution_vng_reordered = new int[n_vn_vng];

        int target_ix;

        for (int i = 0; i < n_vn_vng; i++) {
            int target_p = i % vng_size;
            int p_offset = (n_vn_vng / vng_size) * target_p + std::min(n_vn_vng % vng_size, target_p);
            target_ix = p_offset + i / vng_size;
            Vn_prod_vng[target_ix] = Vn_prod_from_scan_vng[i];
            N_vn_vng_reordered[target_ix] = N_vn_vng[i];
            Vns_distribution_vng_reordered[target_ix] = Vns_distribution_vng[i];
        }
    }
}

void distribute_reordered_vn_prod_vng(double* const Vn_prod_vng, int* const N_vn_vng_reordered, int* const Vns_distribution_vng_reordered, int* const N_vn_of_ps_in_vng, 
        int* const displ_vng_p, double* const Vn_prod_p, int* const N_vn_p, int* const Vns_distribution_proc, const int n_vn_p, const MPI_Comm &vng_comm) {
    MPI_Scatterv(Vn_prod_vng, N_vn_of_ps_in_vng, displ_vng_p, MPI_DOUBLE, Vn_prod_p, n_vn_p, MPI_DOUBLE, 0, vng_comm);
    MPI_Scatterv(N_vn_vng_reordered, N_vn_of_ps_in_vng, displ_vng_p, MPI_INT, N_vn_p, n_vn_p, MPI_INT, 0, vng_comm);
    MPI_Scatterv(Vns_distribution_vng_reordered, N_vn_of_ps_in_vng, displ_vng_p, MPI_INT, Vns_distribution_proc, n_vn_p, MPI_INT, 0, vng_comm);
}

#pragma endregion


#pragma region VNG-masters group

void scatter_vng_sizes(int* const Vng_n_vns, int& n_vn_vng, const MPI_Comm& masters_comm) {
    // inform each master about number of VNs in its VNG

    MPI_Scatter(Vng_n_vns, 1, MPI_INT, &n_vn_vng, 1, MPI_INT, AGDS_MASTER_RANK, masters_comm);
}

void distribute_VN_v_and_VN_n_and_VN_distr(const int rank, const int n_vn_vng, int* const Vng_n_vns, double* const trees, double* const tree,
        int* const N_vn_vngs, int* const N_vn_vng, int* const Vns_distribution, int* const Vns_distribution_vng, const MPI_Comm& masters_comm) {
    // distribute VN_v and VN_n between masters of VNGs

    int displs[N_VNG];
    if (rank == AGDS_MASTER_RANK) {
        cumulated_sum_shifted(Vng_n_vns, N_VNG, displs);
    }

    MPI_Scatterv(trees, Vng_n_vns, displs, MPI_DOUBLE, tree, n_vn_vng, MPI_DOUBLE, AGDS_MASTER_RANK, masters_comm);
    MPI_Scatterv(N_vn_vngs, Vng_n_vns, displs, MPI_INT, N_vn_vng, n_vn_vng, MPI_INT, AGDS_MASTER_RANK, masters_comm);
    MPI_Scatterv(Vns_distribution, Vng_n_vns, displs, MPI_INT, Vns_distribution_vng, n_vn_vng, MPI_INT, AGDS_MASTER_RANK, masters_comm);
}

#pragma endregion


#pragma region World group

// share number of processes belonging to each group (direct and cumulated)
void share_vng_proc_sizes(int* const master_ranks, int* const Vng_n_p) {
    MPI_Bcast(master_ranks, N_VNG, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(Vng_n_p, N_VNG, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);
}

// create group and communicator for masters of VNGs
void setup_vng_communicators(const MPI_Group &world_group, int* const master_ranks, MPI_Group &masters_group, MPI_Comm &masters_comm) {
    MPI_Group_incl(world_group, N_VNG, master_ranks, &masters_group);
    MPI_Comm_create_group(MPI_COMM_WORLD, masters_group, MASTERS_TAG, &masters_comm);
}

void scatter_conn_matrix(const int size, const int rank, int& n_on_p, int* const CONN, int* &CONN_proc, 
        int* const CONN_global_ix, int* &CONN_global_ix_proc) {
    int* CONN_len_all_proc;
    int* displacements;
    if (rank == AGDS_MASTER_RANK) {
        CONN_len_all_proc = new int[size];
        displacements = new int[size];
    }

    n_on_p = N_ON / size + (N_ON % size > rank ? 1 : 0);

    MPI_Gather(&n_on_p, 1, MPI_INT, CONN_len_all_proc, 1, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);

    if (rank == AGDS_MASTER_RANK) {
        for (int i = 0; i < size; i++) {
            CONN_len_all_proc[i] *= N_VNG;
        }
        cumulated_sum_shifted(CONN_len_all_proc, size, displacements);
    }

    CONN_proc = new int[n_on_p * N_VNG];
    CONN_global_ix_proc = new int[n_on_p * N_VNG];

    MPI_Scatterv(CONN, CONN_len_all_proc, displacements, MPI_INT, CONN_proc, n_on_p * N_VNG, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);
    MPI_Scatterv(CONN_global_ix, CONN_len_all_proc, displacements, MPI_INT, CONN_global_ix_proc, n_on_p * N_VNG, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);

    if (rank == AGDS_MASTER_RANK) {
        delete[] CONN_len_all_proc;
        delete[] displacements;
    }
}

void scatter_vn_conns(int* const N_vn_conn, int* const Vns_on_to_vn, int* const Vns_n_conns, int* const proc_offsets, 
        int &n_vn_from_on_conn_proc, int* &Vns_on_to_vn_proc, int* &Vns_n_conns_proc) {
    MPI_Scatter(N_vn_conn, 1, MPI_INT, &n_vn_from_on_conn_proc, 1, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);

    Vns_on_to_vn_proc = new int[n_vn_from_on_conn_proc];
    Vns_n_conns_proc = new int[n_vn_from_on_conn_proc];

    MPI_Scatterv(Vns_on_to_vn, N_vn_conn, proc_offsets, MPI_INT, Vns_on_to_vn_proc, n_vn_from_on_conn_proc, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);
    MPI_Scatterv(Vns_n_conns, N_vn_conn, proc_offsets, MPI_INT, Vns_n_conns_proc, n_vn_from_on_conn_proc, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);
}

#pragma endregion


#pragma region World Master-specific


int desired_n_assigned_connections(const int rank, const int size) {
    const int total_n_connections = N_ON * N_VNG;
    return total_n_connections / size + ((total_n_connections % size > rank) ? 1 : 0);
}


void divide_vn_from_on_conns(const int mpi_size, int n_vn, int* const N_vn_vng, int* const N_vn_conn, int* const Vns_on_to_vn, int* const Vns_distribution, 
        int* const Vns_n_conns, int* const proc_offsets) {
    int proc_ix = 0;
    int conns_assigned_to_proc = 0;
    int conns_to_assign_in_current_vn = N_vn_vng[0];
    int vns_within_proc_ix = 0; // index for filling Vns_on_to_vn and Vns_n_conns arrays

    int desired_n_conns_in_proc;

    N_vn_conn[0] = 0;
    Vns_distribution[0] = 1;
    proc_offsets[0] = 0;

    int vn_ix = 0;

    while (vn_ix < n_vn) {
        desired_n_conns_in_proc = desired_n_assigned_connections(proc_ix, mpi_size);

        if (conns_assigned_to_proc + conns_to_assign_in_current_vn <= desired_n_conns_in_proc) {
            // all connections of VN vn_ix can "fit" into currently filled process

            conns_assigned_to_proc += conns_to_assign_in_current_vn;
            N_vn_conn[proc_ix]++;
            Vns_on_to_vn[vns_within_proc_ix] = vn_ix;
            Vns_n_conns[vns_within_proc_ix] = conns_to_assign_in_current_vn;

            vns_within_proc_ix++;
            
            // moving to next vn
            vn_ix++;
            conns_to_assign_in_current_vn = N_vn_vng[vn_ix];
            Vns_distribution[vn_ix] = 1;
        }
        else {
            if (conns_assigned_to_proc < desired_n_conns_in_proc) {
                // some of the connections of VN vn_ix can "fit" into currently filled process, but some must be transferred to the next process

                N_vn_conn[proc_ix]++;
                Vns_on_to_vn[vns_within_proc_ix] = vn_ix;
                Vns_distribution[vn_ix]++;
                Vns_n_conns[vns_within_proc_ix] = desired_n_conns_in_proc - conns_assigned_to_proc;

                conns_to_assign_in_current_vn -= Vns_n_conns[vns_within_proc_ix];
                vns_within_proc_ix++;
            }
            
            // moving to next process
            conns_assigned_to_proc = 0;
            proc_ix++;

            N_vn_conn[proc_ix] = 0;
            proc_offsets[proc_ix] = vns_within_proc_ix;
        }
    }
}


void get_global_vn_conn_indices(int* const CONN_global_ix, int* const CONN, int* const CONN_local_ix, int* const VN_conn_counts_cumulated, int* const Vng_n_vn_cumulated) {
    int abs_ix;
    
    for (int on_ix = 0; on_ix < N_ON; on_ix++) {
        for (int g_ix = 0; g_ix < N_VNG; g_ix++) {
            abs_ix = on_ix * N_VNG + g_ix;
            CONN_global_ix[abs_ix] = VN_conn_counts_cumulated[CONN[abs_ix] + Vng_n_vn_cumulated[g_ix]] + CONN_local_ix[abs_ix];
        }
    }
}


int* divide_workers(int* counts, int mpi_size, int* masters_indices, int* Vng_n_p) {
    int n_workers = mpi_size;
    int* worker_division = new int[N_VNG];
    int* counts_tmp = new int[N_VNG];
    int total_count, assigned_workers, max_elem_ix;
    bool any_worker_assigned;

    for (int i = 0; i < N_VNG; i++) {
        worker_division[i] = 0;
        counts_tmp[i] = counts[i];
    }

    while (n_workers > 0) {
        total_count = sum(counts, N_VNG);
        any_worker_assigned = false;

        for (int i = 0; i < N_VNG; i++) {
            assigned_workers = n_workers * counts_tmp[i] / total_count;
            worker_division[i] += assigned_workers;
            counts_tmp[i] = counts[i] / (1 + worker_division[i]);
            
            any_worker_assigned |= assigned_workers > 0;
        }

        if (!any_worker_assigned) {
            max_elem_ix = max_elem_index(counts_tmp, N_VNG);
            worker_division[max_elem_ix] += 1;
            counts_tmp[max_elem_ix] = counts[max_elem_ix] / (1 + worker_division[max_elem_ix]);
        }

        n_workers = mpi_size - sum(worker_division, N_VNG);
        any_worker_assigned = false;
    }

    masters_indices[0] = 0;
    for (int g = 1; g < N_VNG; g++) {
        masters_indices[g] = masters_indices[g-1] + worker_division[g-1];
    }

    for (int g = 0; g < N_VNG; g++) {
        Vng_n_p[g] = worker_division[g];
    }

    int* colours = new int[mpi_size];

    int colour = 0;
    for (int i = 0; i < mpi_size; i++) {
        while (worker_division[colour] == 0) {
            colour++;
        }
        colours[i] = colour;
        worker_division[colour]--;
    }

    delete[] worker_division;
    delete[] counts_tmp;

    return colours;
}


int build_tree(double* const values, double* const tree, int* const counts, int* const CONN, int* const CONN_local_ix, const int g_ix) {
    // mock implementation
    double* tree_tmp = new double[N_ON];
    int* counts_tmp = new int[N_ON];
    int distinct_count = 0;
    bool found;

    for (int on_ix = 0; on_ix < N_ON; on_ix++) {
        found = false;

        for (int i = 0; i < distinct_count; i++) {
            if (fabs(tree_tmp[i] - values[on_ix]) < EPSILON) {
                counts_tmp[i]++;
                CONN[on_ix * N_VNG + g_ix] = i;
                CONN_local_ix[on_ix * N_VNG + g_ix] = counts_tmp[i] - 1;
                found = true;
                break;
            }
        }

        if (!found) {
            tree_tmp[distinct_count] = values[on_ix];
            counts_tmp[distinct_count] = 1;
            CONN[on_ix * N_VNG + g_ix] = distinct_count;
            CONN_local_ix[on_ix * N_VNG + g_ix] = 0;
            distinct_count++;
        }
    }

    std::vector<int> indices = sort_indices(tree_tmp, distinct_count); // store ordered indices of VNs in ascending order, i.e. for tree [5., 1., 3.] it will be [1, 2, 0]

    int i_src = 0;
    for (auto i: indices) {
        tree[i_src] = tree_tmp[i];
        counts[i_src] = counts_tmp[i];
        i_src++;
    }

    int* reverse_indices = new int[distinct_count]; // at position i it stores index to which VNi should go in sorted tree, i.e. for tree [5., 1., 3.] it will be [2, 0, 1]
    for (int vn_ix = 0; vn_ix < distinct_count; vn_ix++) {
        reverse_indices[indices[vn_ix]] = vn_ix;
    }

    int absolute_on_ix, new_vn_index;
    for (int on_ix = 0; on_ix < N_ON; on_ix++) {
        absolute_on_ix = on_ix * N_VNG + g_ix;

        new_vn_index = reverse_indices[CONN[absolute_on_ix]];
        CONN[absolute_on_ix] = new_vn_index;
    }

    delete[] tree_tmp;
    delete[] counts_tmp;
    delete[] reverse_indices;

    return distinct_count;
}

// init data, divide processes into groups, build mock tree for each VNG
void setup_data_and_groups(const int size, double* &data, double* const trees, int* const N_vn_vngs, int* const CONN, 
        int* const CONN_global_ix, int* const Vng_n_vns, int* &worker_spread, int* const master_ranks, int* const Vng_n_p, 
        int* const N_vn_conn, int* &Vns_on_to_vn, int* &Vns_distribution, int* &Vns_n_conns, int* const proc_offsets, int &n_vn) {

    int* CONN_local_ix = new int[N_ON * N_VNG];

    n_vn = 0;
    for (int g = 0; g < N_VNG; g++) {
        Vng_n_vns[g] = build_tree(&(data[g * N_ON]), &(trees[n_vn]), &(N_vn_vngs[n_vn]), CONN, CONN_local_ix, g);
        n_vn += Vng_n_vns[g];
        // printf("N_vns_vng_%d: %d\n", g, Vng_n_vns[g]);
    }

    worker_spread = divide_workers(Vng_n_vns, size, master_ranks, Vng_n_p);

    // compute vn-on connections global indices
    int* VN_conn_counts_cumulated = new int[n_vn];
    cumulated_sum_shifted(N_vn_vngs, n_vn, VN_conn_counts_cumulated);

    int* Vng_n_vns_cumulated = new int[N_VNG];
    cumulated_sum_shifted(Vng_n_vns, N_VNG, Vng_n_vns_cumulated);

    get_global_vn_conn_indices(CONN_global_ix, CONN, CONN_local_ix, VN_conn_counts_cumulated, Vng_n_vns_cumulated);

    // divide conns for on->vn step
    Vns_distribution = new int[n_vn];
    Vns_on_to_vn = new int[n_vn + size];
    Vns_n_conns = new int[n_vn + size];

    divide_vn_from_on_conns(size, n_vn, N_vn_vngs, N_vn_conn, Vns_on_to_vn, Vns_distribution, Vns_n_conns, proc_offsets);

    delete[] CONN_local_ix;
    delete[] VN_conn_counts_cumulated;
    delete[] Vng_n_vns_cumulated;
}

#pragma endregion


int main(int argc, char** argv)
{
    #pragma region Init MPI

    MPI_Init(&argc, &argv);

    int size, rank;
    mpi_size_and_rank(MPI_COMM_WORLD, size, rank);

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    MPI_Group masters_group;
    MPI_Comm masters_comm;

    MPI_Comm vng_comm;

    #pragma endregion

    #pragma region Command-line arguments
    double* data;
	
    if (rank == AGDS_MASTER_RANK) {
        if (argc > 1) {
            data = load_data(argv[1], N_ON, N_VNG);

            EPSILON = argc > 2 ? std::stod(argv[2]) : 0.0001;
        }
        else {
            N_ON = 500;
            N_VNG = 4;
            EPSILON = (float)0.0001;

            data = init_full_data(N_VNG, N_ON);
        }
    }

    double constructionStartTime = make_measurement();

	ACTIVATED_VNGS = N_VNG - 1;

    MPI_Bcast(&N_ON, 1, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&N_VNG, 1, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&EPSILON, 1, MPI_DOUBLE, AGDS_MASTER_RANK, MPI_COMM_WORLD);
    MPI_Bcast(&ACTIVATED_VNGS, 1, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);

    #pragma endregion

    #pragma region Init variables

    double* trees;
    int* N_vn_vngs; // number of VN<->ON CONNECTIONS from each VN, stored by VNGs
    int* Vng_n_vns = new int[N_VNG]; // number of VNs in each VNG
    int* worker_spread;
    int master_ranks[N_VNG];
    int Vng_n_p[N_VNG]; // number of processes assigned to each VNG

    int n_vn; // total number of VNs in AGDS

    int* CONN; // all ids of VNs
    int* CONN_proc; // ids of VNs assigned to specific process

    int* CONN_global_ix; // all global ids of ON->VN CONNECTIONS
    int* CONN_global_ix_proc; // global ids of ON->VN CONNECTION held by specific process

    int* N_vn_from_on_conn; // number of distinct VNs assigned to each process in ON->VN
    int* Vns_on_to_vn; // indices of VNs assigned to each process in ON->VN step, stored by processes ([P0_VN0, P0_VN1, ..., P1_VN0, ...])
    int* Vns_n_conns; // number of connections assigned to process for each VN, stored by processes and VNs ([P0_VN0_N, P0_VN1_N, ..., P1_VN0_N])
    int* proc_offsets; // indices of first elemnts in arrays Vns_on_to_vn, Vns_n_conns that are relevant for each process

    int* Vns_distribution; // number of processes over which each VN is "spread" during the ON->VN step
    int* Vns_distribution_vng;
    int* Vns_distribution_vng_reordered;

    int n_vn_from_on_conn_proc;
    int* Vns_on_to_vn_proc;
    int* Vns_distribution_proc;
    int* Vns_n_conns_proc;

    double *values, *tree;
    int *counts, *N_vn_vng;
    int n_vn_vng;
    int n_on_p;

    int vng_id;
    int vng_size, vng_rank;

    double* Vn_prod_from_scan_p;
    double vn_range;
    int n_vn_p;

    int* displ_vng_p;

    int* N_vn_of_ps_in_vng;
    double* Vn_prod_from_scan_vng;
    double* Vn_prod_vng;
    double* Vn_prod_p;

    int* N_vn_vng_reordered;

    int* Vns_vn_to_vn_and_vn_to_on_proc;

    #pragma endregion


    #pragma region Setup

    // setup global Master
    if (rank == AGDS_MASTER_RANK) { // process is a global Master
        trees = new double[N_VNG * N_ON];
        N_vn_vngs = new int[N_VNG * N_ON];
        CONN = new int[N_ON * N_VNG];
        CONN_global_ix = new int[N_ON * N_VNG];
        N_vn_from_on_conn = new int[size];
        proc_offsets = new int[size];

        setup_data_and_groups(size, data, trees, N_vn_vngs, CONN, CONN_global_ix, Vng_n_vns, worker_spread, master_ranks, Vng_n_p, 
            N_vn_from_on_conn, Vns_on_to_vn, Vns_distribution, Vns_n_conns, proc_offsets, n_vn);
    }

    // share basic data with everybody
    share_vng_proc_sizes(master_ranks, Vng_n_p);
    setup_vng_communicators(world_group, master_ranks, masters_group, masters_comm);
    scatter_conn_matrix(size, rank, n_on_p, CONN, CONN_proc, CONN_global_ix, CONN_global_ix_proc);
    scatter_vn_conns(N_vn_from_on_conn, Vns_on_to_vn, Vns_n_conns, proc_offsets, n_vn_from_on_conn_proc, Vns_on_to_vn_proc, Vns_n_conns_proc);

    // setup VNG masters
    if (masters_comm != MPI_COMM_NULL) { // process is a Master of one of the VNGs
        int masters_size, masters_rank;
        mpi_size_and_rank(masters_comm, masters_size, masters_rank);

        scatter_vng_sizes(Vng_n_vns, n_vn_vng, masters_comm);

        tree = new double[n_vn_vng];
        N_vn_vng = new int[n_vn_vng];
        Vns_distribution_vng = new int[n_vn_vng];

        distribute_VN_v_and_VN_n_and_VN_distr(rank, n_vn_vng, Vng_n_vns, trees, tree, N_vn_vngs, N_vn_vng, Vns_distribution, Vns_distribution_vng, masters_comm);
    }

    // setup VNGs
    split_processes_into_vngs(worker_spread, rank, vng_id, vng_size, vng_rank, vng_comm);
    broadcast_number_of_vns_in_own_vng(n_vn_vng, vng_comm);
    compute_vn_prod(tree, n_vn_vng, Vn_prod_from_scan_p, vn_range, n_vn_p, vng_comm);
    broadcast_number_of_vns_in_all_vngs(Vng_n_vns);

    if (vng_rank == 0) {
        N_vn_of_ps_in_vng = new int[vng_size];
        Vn_prod_vng = new double[n_vn_vng];
        Vn_prod_from_scan_vng = new double[n_vn_vng];
    }
    Vn_prod_p = new double[n_vn_p];
    Vns_distribution_proc = new int[n_vn_p];

    gather_number_of_vns_per_proc(n_vn_p, N_vn_of_ps_in_vng, displ_vng_p, vng_rank, vng_size, vng_comm);
    gather_full_vn_prod_from_scan_vng(Vn_prod_from_scan_p, n_vn_p, Vn_prod_from_scan_vng, N_vn_of_ps_in_vng, displ_vng_p, vng_comm);
    reorder_vn_prod_vng(vng_size, vng_rank, n_vn_vng, Vn_prod_vng, Vn_prod_from_scan_vng, N_vn_vng, N_vn_vng_reordered, Vns_distribution_vng, Vns_distribution_vng_reordered);
    int N_vn_p[n_vn_p];
    distribute_reordered_vn_prod_vng(Vn_prod_vng, N_vn_vng_reordered, Vns_distribution_vng_reordered, N_vn_of_ps_in_vng, 
        displ_vng_p, Vn_prod_p, N_vn_p, Vns_distribution_proc, n_vn_p, vng_comm);

    Vns_vn_to_vn_and_vn_to_on_proc = new int[n_vn_p];
    global_vn_indices_vn_to_vn_and_vn_to_on(vng_id, vng_rank, vng_size, n_vn_p, Vng_n_vns, Vns_vn_to_vn_and_vn_to_on_proc);

    #pragma endregion

    #pragma region Experiment
    // compute inferences

    setup_for_inference(N_ON, n_vn_p, n_on_p, N_VNG, Vng_n_p, Vng_n_vns, n_vn_vng, vng_id, vng_comm, CONN_proc, CONN_global_ix_proc, 
        Vn_prod_p, N_vn_p, N_ON * N_VNG, sum(Vns_n_conns_proc, n_vn_from_on_conn_proc), n_vn_from_on_conn_proc, Vns_n_conns_proc, 
        Vns_distribution_proc, Vns_vn_to_vn_and_vn_to_on_proc, Vns_on_to_vn_proc);

    int* all_activated_vns = new int[ACTIVATED_VNS_PER_GROUP * N_VNG];
    int* on_queries = new int[ACTIVATED_ONS * N_QUERIES];

    int n_activated_vns, n_activated_ons;

    double constructionEndTime = make_measurement();
    report_times_expanded("construction", rank, n_vn, size, constructionStartTime, constructionEndTime);

    double start_time, end_time;

    if (K == 1) {
        #pragma region Mock queries

        if (rank == AGDS_MASTER_RANK) {
            srand(SEED);
            mock_on_queries(on_queries, N_QUERIES, N_ON, ACTIVATED_ONS);
        }

        MPI_Bcast(on_queries, ACTIVATED_ONS * N_QUERIES, MPI_INT, AGDS_MASTER_RANK, MPI_COMM_WORLD);

        int local_on_queries[ACTIVATED_ONS * N_QUERIES];
        int local_on_queries_lengths[N_QUERIES];
        int local_on_queries_displacements[N_QUERIES];
        int offset = 0;

        int* n_ons_cumulated = new int[size];
        n_ons_cumulated[0] = n_elems_in_equal_split(N_ON, size, 0);
        for (int proc_ix = 1; proc_ix < size; proc_ix++) {
            n_ons_cumulated[proc_ix] = n_ons_cumulated[proc_ix - 1] + n_elems_in_equal_split(N_ON, size, proc_ix);
        }

        for (int q = 0; q < N_QUERIES; q++) {
            local_on_queries_lengths[q] = 0;
            local_on_queries_displacements[q] = offset;
            for (int i = 0; i < ACTIVATED_ONS; i++) {
                // if (on_queries[q * ACTIVATED_ONS + i] % size == rank) {
                if (upper_bound(n_ons_cumulated, size, on_queries[q * ACTIVATED_ONS + i]) == rank) {
                    local_on_queries[offset] = on_queries[q * ACTIVATED_ONS + i];
                    local_on_queries_lengths[q]++;
                    offset++;
                }
            }
        }

        delete[] n_ons_cumulated;

        #pragma endregion

        // init_logging();
        start_time = make_measurement();

        for (int q_ix = 0; q_ix < N_QUERIES; q_ix++) {
            inference(NULL, 0, &(local_on_queries[local_on_queries_displacements[q_ix]]), local_on_queries_lengths[q_ix], false, K);

            // if (rank == AGDS_MASTER_RANK) {
            //     printf("Finished query %d\n", q_ix);
            // }
        }

        end_time = make_measurement();
        // save_log(size);
    }
    else {
        #pragma region Mock queries

        // generate random inactive (output) VNGs in each query, done at global master
        int vn_queries_inactive_vngs[N_QUERIES * (N_VNG - ACTIVATED_VNGS)];

        if (rank == AGDS_MASTER_RANK) {
            srand(SEED);

            mock_vn_queries_inactive_vngs(vn_queries_inactive_vngs, N_QUERIES, N_VNG, ACTIVATED_VNGS);
        }

        // generate random VNs for queries, done at group masters
        int vn_queries_vng[N_QUERIES * ACTIVATED_VNS_PER_GROUP];
        if (vng_rank == 0) {
            MPI_Bcast(vn_queries_inactive_vngs, N_QUERIES * (N_VNG - ACTIVATED_VNGS), MPI_INT, AGDS_MASTER_RANK, masters_comm);
            
            mock_vn_queries_vng(vn_queries_inactive_vngs, vn_queries_vng, N_VNG, N_QUERIES, ACTIVATED_VNS_PER_GROUP);
        }

        MPI_Bcast(vn_queries_vng, N_QUERIES * ACTIVATED_VNS_PER_GROUP, MPI_INT, 0, vng_comm);

        // find local VNs in queries
        int local_vn_queries[N_QUERIES * ACTIVATED_VNS_PER_GROUP];
        int local_vn_queries_lengths[N_QUERIES];
        int local_vn_queries_displs[N_QUERIES];
        bool local_vn_queries_active[N_QUERIES];
        int offset = 0;

        for (int q = 0; q < N_QUERIES; q++) {
            local_vn_queries_lengths[q] = 0;
            local_vn_queries_displs[q] = offset;

            if (vn_queries_inactive_vngs[2*q] == vng_id || vn_queries_inactive_vngs[2*q + 1] == vng_id) {
                local_vn_queries_active[q] = false;
            }
            else {
                local_vn_queries_active[q] = true;
                for (int i = 0; i < ACTIVATED_VNS_PER_GROUP; i++) {
                    if (on_queries[q * ACTIVATED_VNS_PER_GROUP + i] % vng_size == vng_rank) {
                        local_vn_queries[offset] = vn_queries_vng[q * ACTIVATED_ONS + i];
                        local_vn_queries_lengths[q]++;
                        offset++;
                    }
                }
            }
        }

        #pragma endregion

        // init_logging();
        start_time = make_measurement();

        for (int q_ix = 0; q_ix < N_QUERIES; q_ix++) {
            inference(&(local_vn_queries[local_vn_queries_displs[q_ix]]), local_vn_queries_lengths[q_ix], NULL, 0, local_vn_queries_active[q_ix], K);
        }

        end_time = make_measurement();
        // save_log(size);
    }

    teardown_inference();

    if (argc == 1) {
        report_times(rank, start_time, end_time);
    }
    else {
        report_times_expanded("inference", rank, n_vn, size, start_time, end_time);
    }

    #pragma endregion

    
    # pragma region Cleanup

    delete[] Vng_n_vns;
    delete[] Vn_prod_p;
    delete[] Vn_prod_from_scan_p;
    delete[] CONN_proc;
    delete[] CONN_global_ix_proc;

    delete[] Vns_on_to_vn_proc;
    delete[] Vns_distribution_proc;
    delete[] Vns_n_conns_proc;

    delete[] Vns_vn_to_vn_and_vn_to_on_proc;

    if (rank == AGDS_MASTER_RANK) {
        delete[] data;

        delete[] trees;
        delete[] N_vn_vngs;

        delete[] CONN;
        delete[] CONN_global_ix;

        delete[] N_vn_from_on_conn;
        delete[] Vns_on_to_vn;
        delete[] Vns_distribution;
        delete[] Vns_n_conns;

        delete[] worker_spread;
        delete[] proc_offsets;
    }

    MPI_Group_free(&world_group);
    MPI_Group_free(&masters_group);
    
    if (masters_comm != MPI_COMM_NULL) {
        delete[] tree;
        delete[] N_vn_vng;
        delete[] Vns_distribution_vng;
        delete[] Vns_distribution_vng_reordered;
        delete[] N_vn_of_ps_in_vng;
        delete[] Vn_prod_vng;
        delete[] Vn_prod_from_scan_vng;
        delete[] displ_vng_p;
        delete[] N_vn_vng_reordered;
        MPI_Comm_free(&masters_comm);
    }

    MPI_Comm_free(&vng_comm);

    MPI_Finalize();

    #pragma endregion


    return 0;
}