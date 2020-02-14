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

ThreadPool::ThreadPool(/* args */)
{
    tasks = new queue<Job*>();
    if (sem_init(&sem, 0, 0) == -1)
        handle_error("sem_init");
    if (pthread_mutex_init(&mtx_in, NULL) == -1)
        handle_error("mutex_init");
    if (pthread_mutex_init(&mtx_out, NULL) == -1)
        handle_error("mutex_init");
    // Create threads
    for (int i=0; i<NUM_WORKERS; i++) {
        if(pthread_create(&cThread[i], NULL, (THREADFUNCPTR)&ThreadPool::worker, this)){
            perror("ERROR creating thread.");
        }
    }
}

void ThreadPool::wait_workers_exit()
{
    for (int i=0; i<NUM_WORKERS; i++)
        add_task(NULL);
    for (int i=0; i<NUM_WORKERS; i++)
        pthread_join(cThread[i], NULL);
}

void ThreadPool::wait_queue_empty()
{
    pthread_mutex_lock(&mtx_in);
    pthread_cond_wait(&cond_empty, &mtx_in);
    pthread_mutex_unlock(&mtx_in);
}

void ThreadPool::wait_tasks_complete()
{
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, NUM_WORKERS+1);
    for (int i=0; i<NUM_WORKERS; i++)
        add_task(new BarrierJob(&barrier));
    pthread_barrier_wait(&barrier);
    pthread_barrier_destroy(&barrier);
}

ThreadPool::~ThreadPool()
{
}

void* ThreadPool::worker(void* parm) {
    pthread_t self;
    self = pthread_self();
    ThreadPool *tp = (ThreadPool*)parm;
    while (1)
    {
        Job *pJob = tp->get_task();
        if (pJob == NULL){
            pthread_exit(0);
        }
        pJob->execute();
    }
        
}

void ThreadPool::add_task(Job *pJob)
{
    pthread_mutex_lock(&mtx_in);
    tasks->push(pJob);
    sem_post(&sem);
    pthread_mutex_unlock(&mtx_in);
}

Job* ThreadPool::get_task()
{  
    pthread_mutex_lock(&mtx_out);  // output queue exclusive access
    sem_wait(&sem);
    pthread_mutex_lock(&mtx_in);
    Job* pJob = tasks->front();
    tasks->pop();
    pthread_mutex_unlock(&mtx_in);
    pthread_mutex_unlock(&mtx_out);
    return pJob;
}


#endif // THREAD_POOL_H
