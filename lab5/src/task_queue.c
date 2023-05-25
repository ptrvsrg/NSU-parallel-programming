#include "task.h"
#include <stdlib.h>

struct task_queue_t {
    struct task_t *data;
    int capacity;
    int count;
    int pop_index;
};

struct task_queue_t *task_queue_create(int capacity) {
    struct task_queue_t *queue = malloc(sizeof(struct task_queue_t));
    if (queue == NULL) {
        return NULL;
    }

    struct task_t *data = malloc(sizeof(struct task_t) * capacity);
    if (data == NULL) {
        return NULL;
    }

    queue->data = data;
    queue->capacity = capacity;
    queue->count = 0;
    queue->pop_index = 0;

    return queue;
}

bool task_queue_is_empty(const struct task_queue_t *queue) {
    return queue->count == 0;
}

bool task_queue_is_full(const struct task_queue_t *queue) {
    return queue->count == queue->capacity;
}

int task_queue_push(struct task_queue_t *queue, struct task_t task) {
    if (queue == NULL) {
        return ERROR;
    }

    if (task_queue_is_full(queue)) {
        return ERROR;
    }

    int push_index = (queue->pop_index + queue->count) % queue->capacity;
    queue->data[push_index] = task;
    queue->count++;

    return SUCCESS;
}

int task_queue_pop(struct task_queue_t *queue, struct task_t *task) {
    if (queue == NULL) {
        return ERROR;
    }

    if (task_queue_is_empty(queue)) {
        return ERROR;
    }

    *task = queue->data[queue->pop_index];
    queue->pop_index = (queue->pop_index + 1) % queue->capacity;
    queue->count--;

    return SUCCESS;
}

void task_queue_destroy(struct task_queue_t **queue) {
    if (*queue == NULL) {
        return;
    }

    if ((*queue)->data == NULL) {
        return;
    }

    free((*queue)->data);
    free(*queue);

    *queue = NULL;
}
