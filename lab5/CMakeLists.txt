cmake_minimum_required(VERSION 3.16.3)

project(cluster C)

find_package(MPI REQUIRED)
find_package(Threads REQUIRED)

# Add task queue library
add_library(task_queue STATIC src/task_queue.c)
target_include_directories(task_queue
        PUBLIC include)

add_executable(cluster src/main.c)
target_include_directories(cluster
        PUBLIC include
        PUBLIC ${MPI_C_INCLUDE_DIRS})
target_link_libraries(cluster
        PUBLIC task_queue
        PUBLIC MPI::MPI_C
        PUBLIC Threads::Threads)

# Tests
enable_testing()

# Set environment variable MPI_NP
if(DEFINED ENV{MPI_NP})
    set(MPI_NP $ENV{MPI_NP})
else ()
    set(MPI_NP 2)
endif ()

foreach (MPI_NP RANGE 2 ${MPI_NP})
    add_test(
            NAME test_mpi_np_${MPI_NP}
            COMMAND mpiexec -np ${MPI_NP} ./cluster)
endforeach (MPI_NP)
