#include <mpich/mpi.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "color.h"
#include "task_queue.h"

#define TASK_COUNT              2000
#define REQUEST_TAG             0
#define RESPONSE_TAG            1
#define EMPTY_QUEUE_RESPONSE    (-1)
#define TERMINATION_SIGNAL      (-2)

int process_count;
int process_id;
int proc_sum_weight = 0;
bool termination = false;
struct task_queue_t *task_queue;

pthread_mutex_t mutex;
pthread_cond_t worker_cond;
pthread_cond_t receiver_cond;

void *worker_start(void *args);
void *receiver_start(void *args);
void *sender_start(void *args);

int main(int argc, char *argv[]) {
    int required = MPI_THREAD_MULTIPLE;
    int provided;
    double start_time;
    double end_time;
    pthread_t worker_thread;
    pthread_t receiver_thread;
    pthread_t sender_thread;

    // Initialize MPI environment
    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided != required) {
        return EXIT_FAILURE;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    // Create task queue
    task_queue = task_queue_create(TASK_COUNT);

    // Initialize mutex and condition variable
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&worker_cond, NULL);
    pthread_cond_init(&receiver_cond, NULL);

    // Start worker and sender thread
    start_time = MPI_Wtime();
    pthread_create(&worker_thread, NULL, worker_start, NULL);
    pthread_create(&receiver_thread, NULL, receiver_start, NULL);
    pthread_create(&sender_thread, NULL, sender_start, NULL);

    pthread_join(worker_thread, NULL);
    pthread_join(receiver_thread, NULL);
    pthread_join(sender_thread, NULL);
    end_time = MPI_Wtime();

    // Print result
    MPI_Barrier(MPI_COMM_WORLD);
    printf(FGREEN"Summary weight %d: %lf\n"FNORM, process_id, proc_sum_weight * 1E-6);
    MPI_Barrier(MPI_COMM_WORLD);
    if (process_id == 0) {
        printf(FGREEN"Time: %lf\n"FNORM, end_time - start_time);
    }

    // Clear resources
    task_queue_destroy(&task_queue);
    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&worker_cond);
    pthread_cond_destroy(&receiver_cond);
    MPI_Finalize();

    return EXIT_SUCCESS;
}

static inline void init_tasks() {
    // Total sum of task weights does not change
    // For each process, tasks have a weight: min_weight * (process_id + 1)
    // min_weight = (TOTAL_SUM_WEIGHT * n) / (TASK_COUNT * S_n)
    // n - process count
    // S_n - sum of an arithmetic progression 1, 2, ... , n

    const int TOTAL_SUM_WEIGHT = 50000000;
    int min_weight = 2 * TOTAL_SUM_WEIGHT / (TASK_COUNT * (process_count + 1));
    int task_id = 1;

    for (int i = 0; i < TASK_COUNT; ++i) {
        // Create task
        struct task_t task = {
            .id = task_id,
            .process_id = process_id,
            .weight = min_weight * (i % process_count + 1)
        };

        if (i % process_count == process_id) {
            task_queue_push(task_queue, task);
            task_id++;
            proc_sum_weight += task.weight;
        }
    }
}

static inline void execute_tasks() {
    while (true) {
        struct task_t task;

        pthread_mutex_lock(&mutex);
        if (task_queue_is_empty(task_queue)) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        task_queue_pop(task_queue, &task);
        pthread_mutex_unlock(&mutex);

        printf(FBLUE"Worker %d executing task %d of process %d and weight %d\n"FNORM,
               process_id,
               task.id,
               task.process_id,
               task.weight);
        usleep(task.weight);
    }
}

void *worker_start(void *args) {
    init_tasks();

    // Worker start synchronization
    MPI_Barrier(MPI_COMM_WORLD);

    while (true) {
        execute_tasks();

        pthread_mutex_lock(&mutex);
        while (task_queue_is_empty(task_queue) && !termination) {
            pthread_cond_signal(&receiver_cond);
            pthread_cond_wait(&worker_cond, &mutex);
        }

        if (termination) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        pthread_mutex_unlock(&mutex);
    }

    printf(FBLUE"Worker %d finished\n"FNORM, process_id);
    pthread_exit(NULL);
}

void *receiver_start(void *args) {
    int termination_signal = TERMINATION_SIGNAL;

    while (!termination) {
        int received_tasks = 0;
        struct task_t task;

        pthread_mutex_lock(&mutex);
        while (!task_queue_is_empty(task_queue)) {
            pthread_cond_wait(&receiver_cond, &mutex);
        }
        pthread_mutex_unlock(&mutex);

        for (int i = 0; i < process_count; ++i) {
            if (i == process_id) {
                continue;
            }

            printf(FYELLOW"Receiver %d sent request to process %d\n"FNORM, process_id, i);
            MPI_Send(&process_id, 1, MPI_INT, i, REQUEST_TAG, MPI_COMM_WORLD);
            MPI_Recv(&task, sizeof(task), MPI_BYTE, i, RESPONSE_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (task.id != EMPTY_QUEUE_RESPONSE) {
                printf(FYELLOW"Receiver %d received task %d from process %d\n"FNORM, process_id, task.id, i);

                pthread_mutex_lock(&mutex);
                task_queue_push(task_queue, task);
                pthread_mutex_unlock(&mutex);

                received_tasks++;
            } else {
                printf(FYELLOW"Receiver %d received empty queue response from process %d\n"FNORM, process_id, i);
            }
        }

        if (received_tasks == 0) {
            pthread_mutex_lock(&mutex);
            termination = true;
            pthread_mutex_unlock(&mutex);
        }

        pthread_mutex_lock(&mutex);
        pthread_cond_signal(&worker_cond);
        pthread_mutex_unlock(&mutex);
    }

    // Receiver destruction synchronization
    MPI_Barrier(MPI_COMM_WORLD);

    printf(FYELLOW"Receiver %d sent termination signal\n"FNORM, process_id);
    MPI_Send(&termination_signal, 1, MPI_INT, process_id, REQUEST_TAG, MPI_COMM_WORLD);

    printf(FYELLOW"Receiver %d finished\n"FNORM, process_id);
    pthread_exit(NULL);
}

void *sender_start(void *args) {
    while (true) {
        int receive_process_id;
        struct task_t task;

        printf(FMAGENTA"Sender %d waiting for request\n"FNORM, process_id);
        MPI_Recv(&receive_process_id, 1, MPI_INT, MPI_ANY_SOURCE, REQUEST_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (receive_process_id == TERMINATION_SIGNAL) {
            printf(FMAGENTA"Sender %d received termination signal\n"FNORM, process_id);
            break;
        }

        printf(FMAGENTA"Sender %d received request from process %d\n"FNORM, process_id, receive_process_id);

        pthread_mutex_lock(&mutex);
        if (!task_queue_is_empty(task_queue)) {
            task_queue_pop(task_queue, &task);
            printf(FMAGENTA"Sender %d sent task %d of process %d to process %d\n"FNORM,
                   process_id,
                   task.id,
                   task.process_id,
                   receive_process_id);
        } else {
            task.id = EMPTY_QUEUE_RESPONSE;
            task.weight = 0;
            task.process_id = process_id;
            printf(FMAGENTA"Sender %d sent empty queue response to process %d\n"FNORM, process_id, receive_process_id);
        }
        pthread_mutex_unlock(&mutex);

        MPI_Send(&task, sizeof(task), MPI_BYTE, receive_process_id, RESPONSE_TAG, MPI_COMM_WORLD);
    }

    printf(FMAGENTA"Sender %d finished\n"FNORM, process_id);
    pthread_exit(NULL);
}