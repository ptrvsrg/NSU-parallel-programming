#ifndef TASK_H
#define TASK_H

#include <stdbool.h>

#define SUCCESS 0
#define ERROR   (-1)

struct task_t {
    int id;
    int process_id;
    int weight;
};

struct task_queue_t;

struct task_queue_t *task_queue_create(int capacity);
bool task_queue_is_empty(const struct task_queue_t *queue);
bool task_queue_is_full(const struct task_queue_t *queue);
int task_queue_push(struct task_queue_t *queue, struct task_t task);
int task_queue_pop(struct task_queue_t *queue, struct task_t *task);
void task_queue_destroy(struct task_queue_t **queue);

#endif // TASK_H
