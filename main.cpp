#include <iostream>
#include <mpi.h>
#include <ctime>

double get_rand(double min, double max) {
    srand(time(0) + rand());

    double num = min + static_cast <double> (rand()) / (static_cast <double> (RAND_MAX / (max - min)));

    return num;
}

void randomize_matrix(double *matrix, int size) {
    for (int i = 0; i < size; ++i) {
        matrix[i] = get_rand(0.0, 10.0);
    }
}

double mult_arrays(const double *arr1, const double *arr2, int size) {
    double result = 0;

    for (int i = 0; i < size; ++i) {
        result += arr1[i] * arr2[i];
    }

    return result;
}

void define_send_count(int *send_counts, int size, int vector_length) {
    for (int i = 0; i < size; ++i) {
        send_counts[i] = vector_length / size;

        if (i < vector_length % size) {
            send_counts[i] += 1;
        }
    }
}

void define_displacement(int *displs, const int *send_counts, int size) {
    int current_displacement = 0;
    for (int i = 0; i < size; ++i) {
        displs[i] = current_displacement;
        current_displacement += send_counts[i];
    }
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int size, rank, root = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int VECTORS_LENGTH = 1000000;
    double *vector_x = new double[VECTORS_LENGTH];
    double *vector_y = new double[VECTORS_LENGTH];
    int *sendcounts = new int[size];
    int *displs = new int[size];
    double time_start;


    if (rank == root) time_start = clock();


    for (int i = 0; i < size; ++i) {
        define_send_count(sendcounts, size, VECTORS_LENGTH);
        define_displacement(displs, sendcounts, size);
    }

    double *local_vector_x = new double[sendcounts[rank]];
    double *local_vector_y = new double[sendcounts[rank]];


    MPI_Scatterv(vector_x, sendcounts, displs, MPI_DOUBLE, local_vector_x, sendcounts[rank], MPI_DOUBLE, root,
                 MPI_COMM_WORLD);
    MPI_Scatterv(vector_y, sendcounts, displs, MPI_DOUBLE, local_vector_y, sendcounts[rank], MPI_DOUBLE, root,
                 MPI_COMM_WORLD);

    randomize_matrix(local_vector_x, sendcounts[rank]);
    randomize_matrix(local_vector_y, sendcounts[rank]);

    // local scalar product
    double local_result = mult_arrays(local_vector_x, local_vector_y, sendcounts[rank]);

    // summation of local results
    double result;
    MPI_Reduce(&local_result, &result, 1, MPI_DOUBLE, MPI_SUM, root, MPI_COMM_WORLD);

    if (rank == root) {
        // parallel calculations summary
        printf("Vector size: %dm\n", VECTORS_LENGTH / 1000000);
        printf("Processes: %d\n", size);
        printf("Time taken: %.2fs\n", (double) (clock() - time_start) / CLOCKS_PER_SEC);
    }


    MPI_Finalize();
}
