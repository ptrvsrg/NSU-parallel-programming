#include <mpich/mpi.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "color.h"
#include "task_queue.h"

#define LISTS_COUNT             2
#define QUEUE_CAPACITY          10
#define REQUEST_TAG             0
#define RESPONSE_TAG            1
#define EMPTY_QUEUE_RESPONSE    (-1)
#define TERMINATION_SIGNAL      (-2)

int process_count;
int process_id;
int sum_weight = 0;
struct task_queue_t *task_queue;
pthread_mutex_t mutex;

void init_tasks();
void execute_tasks();
void* worker_start(void* args);
void* sender_start(void* args);

int main(int argc, char* argv[]) {
    int required = MPI_THREAD_MULTIPLE;
    int provided;
    double start_time;
    double end_time;
    pthread_t worker_thread;
    pthread_t sender_thread;

    // Initialize MPI environment
    MPI_Init_thread(&argc, &argv, required, &provided);
    if(provided != required) {
        return EXIT_FAILURE;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    // Create task queue
    task_queue = task_queue_create(QUEUE_CAPACITY);

    // Initialize mutex
    pthread_mutex_init(&mutex, NULL);

    // Start worker and sender thread
    start_time = MPI_Wtime();
    pthread_create(&worker_thread, NULL, worker_start, NULL);
    pthread_create(&sender_thread, NULL, sender_start, NULL);

    pthread_join(worker_thread, NULL);
    pthread_join(sender_thread, NULL);
    end_time = MPI_Wtime();

    // Print result
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Sum weight: %lf\n", sum_weight * 1E-6);
    MPI_Barrier(MPI_COMM_WORLD);
    if (process_id == 0) {
        printf("Time: %lf\n", end_time - start_time);
    }

    // Clear resources
    pthread_mutex_destroy(&mutex);
    task_queue_destroy(&task_queue);
    MPI_Finalize();

    return EXIT_SUCCESS;
}

void init_tasks() {
    int task_id = 1;
    unsigned int seed = time(NULL) + process_id;

    while (!task_queue_is_full(task_queue)) {
        struct task_t task = {
            .id = task_id,
            .process_id = process_id,
            .weight = (30000 + rand_r(&seed) % 30000) * (process_id + 1)
        };
        task_queue_push(task_queue, task);

        task_id++;
        sum_weight += task.weight;
    }
}

void execute_tasks() {
    struct task_t task;

    while (true) {
        pthread_mutex_lock(&mutex);
        if (task_queue_is_empty(task_queue)) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        task_queue_pop(task_queue, &task);
        pthread_mutex_unlock(&mutex);

        printf(FBLUE"Worker-Receiver %d executing task %d of process %d and weight %d\n"FNORM,
               process_id,
               task.id,
               task.process_id,
               task.weight);
        usleep(task.weight);
    }
}

void* worker_start(void* args) {
    struct task_t task;

    for(int i = 0; i < LISTS_COUNT; i++) {
        MPI_Barrier(MPI_COMM_WORLD);

        init_tasks();
        execute_tasks();

        while (true) {
            int received_tasks = 0;
            for (int j = 0; j < process_count; j++) {
                if (j == process_id) {
                    continue;
                }

                printf(FYELLOW"Worker-Receiver %d sent request to process %d\n"FNORM, process_id, j);
                MPI_Send(&process_id, 1, MPI_INT, j, REQUEST_TAG, MPI_COMM_WORLD);

                MPI_Recv(&task, sizeof(task), MPI_BYTE, j, RESPONSE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                if (task.id != EMPTY_QUEUE_RESPONSE) {
                    printf(FYELLOW"Worker-Receiver %d receive task %d from process %d\n"FNORM, process_id, task.id, j);
                    pthread_mutex_lock(&mutex);
                    task_queue_push(task_queue, task);
                    pthread_mutex_unlock(&mutex);

                    execute_tasks();

                    received_tasks++;
                } else {
                    printf(FYELLOW"Worker-Receiver %d receive empty queue task from process %d\n"FNORM, process_id, j);
                }
            }

            if (received_tasks == 0) {
                break;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    int termination_signal = TERMINATION_SIGNAL;
    printf(FYELLOW"Worker-Receiver %d sent execution signal\n"FNORM, process_id);
    MPI_Send(&termination_signal, 1, MPI_INT, process_id, REQUEST_TAG, MPI_COMM_WORLD);

    printf(FMAGENTA"Worker-Receiver %d finish\n"FNORM, process_id);
    pthread_exit(NULL);
}

void* sender_start(void* args) {
    int receive_process_id;
    struct task_t task;
    MPI_Status status;
    MPI_Barrier(MPI_COMM_WORLD);
    while (true) {
        printf(FMAGENTA"Sender %d waiting for request\n"FNORM, process_id);
        MPI_Recv(&receive_process_id, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, &status);

        if (receive_process_id == TERMINATION_SIGNAL) {
            printf(FMAGENTA"Sender %d received request from process %d\n"FNORM, process_id, receive_process_id);
            break;
        }

        printf(FMAGENTA"Sender %d received request from process %d\n"FNORM, process_id, receive_process_id);

        pthread_mutex_lock(&mutex);
        if (!task_queue_is_empty(task_queue)){
            task_queue_pop(task_queue, &task);
            printf(FMAGENTA"Sender %d send task %d of process %d to process %d\n"FNORM,
                   process_id,
                   task.id,
                   task.process_id,
                   receive_process_id);
        } else {
            task.id = EMPTY_QUEUE_RESPONSE;
            task.weight = 0;
            task.process_id = process_id;
            printf(FMAGENTA"Sender %d send empty queue response to process %d\n"FNORM, process_id, receive_process_id);
        }
        pthread_mutex_unlock(&mutex);

        MPI_Send(&task, sizeof(task), MPI_BYTE, receive_process_id, RESPONSE_TAG, MPI_COMM_WORLD);
    }

    printf(FMAGENTA"Sender %d finish\n"FNORM, process_id);
    pthread_exit(NULL);
}