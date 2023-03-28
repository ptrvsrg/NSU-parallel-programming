/**
 * @file mpi_multiply.c
 * @author ptrvsrg (s.petrov1@g.nsu.ru)
 * @brief Multiplication of matrices by division into blocks using MPI
 * @version 0.1
 */

#include <math.h>
#include <mpi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define N_1 3
#define N_2 4
#define N_3 2
#define DIMS_COUNT 2
#define X 0
#define Y 1

void generate_matrix(double* matrix, int column, int leading_row, int leading_column, bool onRows);
void split_A(const double* A, double* A_block, int A_block_size, int coords_y, MPI_Comm comm_rows, MPI_Comm comm_columns);
void split_B(const double* B, double* B_block, int B_block_size, int coords_x, MPI_Comm comm_rows, MPI_Comm comm_columns);
void multiply(const double* A_block, const double* B_block, double* C_block, int A_block_size, int B_block_size);
void gather_C(const double* C_block, double* C, int A_block_size, int B_block_size, int proc_rank, MPI_Comm comm_grid);
bool check_C(const double* C);
void print_matrix(const double* matrix, int column, int leading_row, int leading_column);

int main(int argc, char **argv)
{
    int proc_rank;
    int proc_count;
    int reorder = 1;
    int aligned_n1;
    int aligned_n3;
    int A_block_size;
    int B_block_size;
    int dims[DIMS_COUNT] = {};
    int periods[DIMS_COUNT] = {};
    int coords[DIMS_COUNT] = {};
    double* A = NULL;
    double* B = NULL;
    double* C = NULL;
    double* A_block = NULL;
    double* B_block = NULL;
    double* C_block = NULL;
    MPI_Comm comm_grid;
    MPI_Comm comm_rows;
    MPI_Comm comm_columns;

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

    // Create division of processes in cartesian grid
    MPI_Dims_create(proc_count, DIMS_COUNT, dims);

    // Make new communicator for blocks, rows and columns
    MPI_Cart_create(MPI_COMM_WORLD, DIMS_COUNT, dims, periods, reorder, &comm_grid);
    MPI_Comm_split(comm_grid, coords[X], coords[Y], &comm_columns);
    MPI_Comm_split(comm_grid, coords[Y], coords[X], &comm_rows);

    // Get coordinates of processes
    MPI_Cart_coords(comm_grid, proc_rank, DIMS_COUNT, coords);

    // Set parameters of matrix blocks
    A_block_size = ceil((double) N_1 / dims[X]);
    B_block_size = ceil((double) N_3 / dims[Y]);
    aligned_n1 = A_block_size * dims[X];
    aligned_n3 = B_block_size * dims[Y];

    // Generate matrix A and B
    if (coords[X] == 0 && coords[Y] == 0)
    {
        A = malloc(sizeof(double) * aligned_n1 * N_2);
        B = malloc(sizeof(double) * N_2 * aligned_n3);
        C = malloc(sizeof(double) * aligned_n1 * aligned_n3);

        generate_matrix(A, N_2, N_1, N_2, true);
        generate_matrix(B, aligned_n3, N_2, N_3, false);

            printf("A:\n");
            print_matrix(A, N_2, N_1, N_2);
            printf("B:\n");
            print_matrix(B, aligned_n3, N_2, N_3);
    }

        MPI_Barrier(comm_grid);

    A_block = malloc(sizeof(double) * A_block_size * N_2);
    B_block = malloc(sizeof(double) * B_block_size * N_2);
    C_block = malloc(sizeof(double) * A_block_size * B_block_size);

    // Split matrix A, B
    split_A(A, A_block, A_block_size, coords[Y], comm_rows, comm_columns);
    split_B(B, B_block, B_block_size, coords[X], comm_rows, comm_columns);

    // multiply matrices A and B
    multiply(A_block, B_block, C_block, A_block_size,  B_block_size);

    // Check matrix C
    if (coords[Y] == 0 && coords[X] == 0)
    {
        printf("Matrix C:\n");
        print_matrix(C, aligned_n3, N_1, N_3);

        // printf("Is matrix C correct? - %s\n", 
        //        check_C(C, N_1, N_3, 1.0 * 2.0 * N_2) ? "yes" : "no");
        
        free(A);
        free(B);
        free(C);
    }

    free(A_block);
    free(B_block);
    free(C_block);

    MPI_Finalize();

    return EXIT_SUCCESS;
}

/**
 * @brief Generate matrix that has same numbers on rows or columns
 * 
 * @param matrix Pointer to array of row*column size
 * @param columnn Number of column
 * @param leading_row Number of rows in which data will be stored (leading_row <= row)
 * @param leading_column Number of columns in which data will be stored (leading_column <= column)
 * @param onRows true - rows have same numbers, false - columns have same numbers
 */
void generate_matrix(double* matrix, int column, int leading_row, int leading_column, bool onRows)
{
    for (int i = 0; i < leading_row; ++i)
        for (int j = 0; j < leading_column; ++j)
            matrix[i * column + j] = onRows ? i : j;
}

/**
 * @brief Split matrix A into row blocks between processes
 * 
 * @param A Matrix A which is only available in 0 process
 * @param A_block Row block
 * @param A_block_size Number of rows in row block
 * @param coords_y Y coordinate of process
 * @param comm_rows Row communicator
 * @param comm_columns Column communicator
 */
void split_A(const double* A, double* A_block, int A_block_size, int coords_y, MPI_Comm comm_rows, MPI_Comm comm_columns)
{
    if (coords_y == 0)
    {
        MPI_Datatype row_t;

        MPI_Type_contiguous(N_2, MPI_DOUBLE, &row_t);
        MPI_Type_commit(&row_t);

        MPI_Scatter(A, A_block_size, row_t, A_block, A_block_size, row_t, 0, comm_rows);

        MPI_Type_free(&row_t);
    }

    MPI_Bcast(A_block, A_block_size * N_2, MPI_DOUBLE, 0, comm_columns);
}

/**
 * @brief Split matrix B into column blocks between processes
 * 
 * @param B Matrix B which is only available in 0 process
 * @param B_block Column block
 * @param B_block_size Number of columns in column block
 * @param coords_x X coordinate of process
 * @param comm_rows Row communicator
 * @param comm_columns Column communicator
 */
void split_B(const double* B, double* B_block, int B_block_size, int coords_x, MPI_Comm comm_rows, MPI_Comm comm_columns)
{
    if (coords_x == 0) 
    {
        MPI_Datatype band_t;
        MPI_Datatype column_not_resized_t;
        MPI_Datatype column_resized_t;

        MPI_Type_contiguous(N_2, MPI_DOUBLE, &band_t);
        MPI_Type_commit(&band_t);

        MPI_Type_vector(N_2, 1, N_3, MPI_DOUBLE, &column_not_resized_t);
        MPI_Type_create_resized(column_not_resized_t, 0, sizeof(double), &column_resized_t);
        MPI_Type_commit(&column_resized_t);

        MPI_Scatter(B, B_block_size, column_resized_t, B_block, B_block_size, band_t, 0, comm_columns);

        MPI_Type_free(&band_t);
        MPI_Type_free(&column_not_resized_t);
        MPI_Type_free(&column_resized_t);
    }

    MPI_Bcast(B_block, B_block_size * N_2, MPI_DOUBLE, 0, comm_rows);
}

/**
 * @brief Multiply row block of matrix A and column block of matrix B
 * 
 * @param A_block Row block of matrix A
 * @param B_block Column block of matrix B
 * @param C_block Grid block of matrix C
 * @param A_block_size Number of rows in row block
 * @param B_block_size Number of columns in column block
 */
void multiply(const double* A_block, const double* B_block, double* C_block, int A_block_size, int B_block_size)
{
    for (int i = 0; i < A_block_size; ++i)
        for (int j = 0; j < B_block_size; ++j)
                C_block[i * B_block_size + j] = 0;

    for (int i = 0; i < A_block_size; ++i)
        for (int j = 0; j < N_2; ++j)
            for (int k = 0; k < B_block_size; ++k)
                C_block[i * B_block_size + k] += A_block[i * N_2 + j] * B_block[j * B_block_size + k];
}

/**
 * @brief Gather matrix C of grid blocks
 * 
 * @param C_block Grid block of matrix C
 * @param C Matrix C
 * @param A_block Row block of matrix A
 * @param B_block Column block of matrix B
 * @param proc_rank Rank of current process 
 * @param comm_grid Grid communicator
 */
void gather_C(const double* C_block, double* C, int A_block_size, int B_block_size, int proc_rank, MPI_Comm comm_grid)
{
    MPI_Datatype send_t;
    MPI_Datatype not_resized_recv_t;
    MPI_Datatype resized_recv_t;

    MPI_Type_contiguous(A_block_size * B_block_size, MPI_DOUBLE, &send_t);
    MPI_Type_commit(&send_t);

    MPI_Type_vector(B_block_size, A_block_size, N_3, MPI_DOUBLE, &not_resized_recv_t);
    MPI_Type_commit(&resized_recv_t);

    MPI_Type_create_resized(not_resized_recv_t, 0, A_block_size * sizeof(double), &resized_recv_t);
    MPI_Type_commit(&resized_recv_t);

    // TODO: MPI_Gatherv

    MPI_Type_free(&send_t);
    MPI_Type_free(&not_resized_recv_t);
    MPI_Type_free(&resized_recv_t);
}

/**
 * @brief Сheck result of multiplying matrices A and B stored in matrix C for correctness
 * 
 * @param C matrix C storing rsult of multiplying matrices A and B
 * @return true - Result of multiplying matrices A and B stored in matrix C is correct,
 * @return false - Result of multiplying matrices A and B stored in matrix C is incorrect
 */
bool check_C(const double* C)
{
    for (int i = 0; i < N_1; ++i)
        for (int j = 0; j < N_3; ++j)
            if (C[i * N_3 + j] != i * j * N_2)
                return false;

    return true;
}

/**
 * @brief Print matrix
 * 
 * @param matrix Pointer to array of row*column size
 * @param column Number of column
 * @param leading_row Number of rows in which data will be stored (leading_row <= row)
 * @param leading_column Number of columns in which data will be stored (leading_column <= column)
 */
void print_matrix(const double* matrix, int column, int leading_row, int leading_column)
{
    for (int i = 0; i < leading_row; i++)
    {
        for (int j = 0; j < leading_column; j++)
            printf("%lf ", matrix[i * column + j]);

        printf("\n");
    }
}
