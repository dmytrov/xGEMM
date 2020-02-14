#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <iostream>
#include <cstring>
#include <semaphore.h>
#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <queue> 
#include <pthread.h>
#include <unistd.h>

using namespace std; 

#define handle_error(msg) \
    do { perror(msg); exit(EXIT_FAILURE); } while (0)

#define NUM_WORKERS 4

class Job
{
public:
    virtual void execute(){};
};


class EmptyJob: public Job
{
public:
    EmptyJob(){
    }

    void execute() override {
    };
};


class PrintJob: public Job
{
public:
    int i;

    PrintJob(int _i){
        i = _i;
    }

    void execute() override {
        std::cout << i << std::endl;
    };
};


class BarrierJob: public Job
{
    pthread_barrier_t *pbarrier;

public:
    BarrierJob(pthread_barrier_t *_pbarrier) {
        pbarrier = _pbarrier;
    }

    void execute() override {
        pthread_barrier_wait(pbarrier);
    };
};


class ThreadPool
{
    //  Create N threads for N CPUs.
    //  Task queue is controlled by an access mutex
    //  Task queue creates an event 
private:
    /* data */
    sem_t sem;
    pthread_mutex_t mtx_in;
    pthread_mutex_t mtx_out;
    pthread_cond_t cond_empty;
    queue<Job*> *tasks;
    pthread_t cThread[NUM_WORKERS];
public:
    ThreadPool(/* args */);
    ~ThreadPool();

    void add_task(Job *pJob);
    Job* get_task();
    static void *worker(void *parm);
    void wait_queue_empty();
    void wait_tasks_complete();
    void wait_workers_exit();
};

typedef void *(*THREADFUNCPTR)(void *);


#endif // THREAD_POOL_H
