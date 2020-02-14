#include "thread_pool.h"

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
