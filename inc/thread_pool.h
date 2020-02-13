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
    queue<int> *tasks;
    pthread_t cThread[NUM_WORKERS];
public:
    ThreadPool(/* args */);
    ~ThreadPool();

    void add_task(int i);
    int get_task();
    static void *worker(void *parm);
    void wait_workers_exit();
};

typedef void *(*THREADFUNCPTR)(void *);

ThreadPool::ThreadPool(/* args */)
{
    tasks = new queue<int>();
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
        add_task(-1);
    for (int i=0; i<NUM_WORKERS; i++)
        pthread_join(cThread[i], NULL);
}

ThreadPool::~ThreadPool()
{
}

void* ThreadPool::worker(void* parm) {
    pthread_t self;
    self = pthread_self();
    ThreadPool *tp = (ThreadPool*)parm;
    std::cout << "Process " << self << " starting" << std::endl;
    while (1)
    {
        int i = tp->get_task();
        //std::cout << "Process " << self << " " << i << " " << std::endl;
        if (i == -1){
            std::cout << "Process " << self << " exiting" << std::endl;
            return NULL;
        }
            
    }
        
}

void ThreadPool::add_task(int i)
{
    pthread_mutex_lock(&mtx_in);
    //std::cout << "Put " << i << std::endl;
    tasks->push(i);
    sem_post(&sem);
    pthread_mutex_unlock(&mtx_in);
}

int ThreadPool::get_task()
{  
    pthread_mutex_lock(&mtx_out);
    sem_wait(&sem);
    pthread_mutex_lock(&mtx_in);
    int i = tasks->front();
    //std::cout << "Pop " << i << std::endl;
    tasks->pop();
    pthread_mutex_unlock(&mtx_in);
    pthread_mutex_unlock(&mtx_out);
    return i;
}


#endif // THREAD_POOL_H
