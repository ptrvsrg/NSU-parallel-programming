cmake_minimum_required(VERSION 3.16.3)

project(mpi_multiply C)

find_package(MPI REQUIRED)

add_executable(mpi_multiply src/mpi_multiply.c)
target_include_directories(mpi_multiply
        PUBLIC ${MPI_C_INCLUDE_DIRS})
target_link_libraries(mpi_multiply
        PUBLIC MPI::MPI_C
        PUBLIC m)

# Tests
enable_testing()
set(MPI_NP 1 2 4 8 16)
set(DIMS_X 1 2 4 8 16)

# Tests for default grid
foreach (MPI_NP ${MPI_NP})
    add_test(
            NAME test_default_grid_mpi_np_${MPI_NP}
            COMMAND mpiexec -np ${MPI_NP} ./mpi_multiply)
endforeach (MPI_NP)

# Tests for custom grid
foreach (DIMS_X ${DIMS_X})
    math(EXPR DIMS_Y "16 / ${DIMS_X}")
    add_test(
            NAME test_custom_grid_${DIMS_X}_${DIMS_Y}
            COMMAND mpiexec -np 16 ./mpi_multiply ${DIMS_X} ${DIMS_Y})
endforeach (DIMS_X)
