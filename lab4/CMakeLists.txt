cmake_minimum_required(VERSION 3.16.3)

project(mpi_jacobi C)

find_package(MPI REQUIRED)

# Add version 1
add_executable(mpi_jacobi_1 src/mpi_jacobi_1.c)
target_include_directories(mpi_jacobi_1
        PUBLIC ${MPI_C_INCLUDE_DIRS})
target_link_libraries(mpi_jacobi_1
        PUBLIC MPI::MPI_C
        PUBLIC m)

# Add version 2
add_executable(mpi_jacobi_2 src/mpi_jacobi_2.c)
target_include_directories(mpi_jacobi_2
        PUBLIC ${MPI_C_INCLUDE_DIRS})
target_link_libraries(mpi_jacobi_2
        PUBLIC MPI::MPI_C
        PUBLIC m)

# Tests
enable_testing()
set(MPI_NP 1 2 4 8 16)

# Tests for version 1
foreach (MPI_NP ${MPI_NP})
    add_test(
            NAME test_version_1_mpi_np_${MPI_NP}
            COMMAND mpiexec -np ${MPI_NP} ./mpi_jacobi_1)
endforeach (MPI_NP)

# Tests for version 2
foreach (MPI_NP ${MPI_NP})
    add_test(
            NAME test_version_2_mpi_np_${MPI_NP}
            COMMAND mpiexec -np ${MPI_NP} ./mpi_jacobi_2)
endforeach (MPI_NP)
