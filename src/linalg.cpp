
#include "thread_pool.h"
#include "linalg.h"

ThreadPool tp;


void MMfloatJob::execute()
{
    execute_unoptimized();
}

void MMfloatJob::execute_unoptimized()
{
    if (a->d1 != b->d0)
        throw std::out_of_range("Matrix dinemsions do not agree.");
    for (int i=ar0; i<ar1; i++) {
        for (int j=bc0; j<bc1; j++){
            float s = 0;
            for (int k=0; k< a->d1; k++){
                s += a->getitem(i, k) * b->getitem(k, j);
            }
            c->setitem(i, j, s);
        }
    }
}

void MMfloatJob::execute_optimized()
{

}


void MMfloat(Mfloat *a, Mfloat *b, Mfloat *c)
{
    // Partition on multiple independent tasks
    tp.add_task(new MMfloatJob(a, 0, a->d0, b, 0, b->d1, c));
    tp.wait_tasks_complete();
}