#include <immintrin.h>
#include "thread_pool.h"
#include "linalg.h"

ThreadPool tp;


void MMfloatJob::execute()
{
    execute_optimized();
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
    // Use 8 columns to stay within chache
    if (bc1-bc0 != 8)
        throw std::invalid_argument("Number of columns must be 8");

    

    for (int i=ar0; i<ar1; i++) {  // rows of a
        float *ai = a->data + i*a->s0;  // a[i, :] row
        
        float acc[8] __attribute__((aligned(32))); // accumulator. Aligned
        std::memset(acc, 0, 8*sizeof(float));
        
        for (int j=bc0; j<bc1; j++) {  // columns of b
            float *bj = b->data + j*b->s0;

            for (int k=0; k<a->d1; k+=8) {  // items
                float *pa = ai+k;
                float *pb = bj+k;
                acc[j-bc0] +=   ai[0] * bj[0] +
                                ai[1] * bj[1] +
                                ai[2] * bj[2] +
                                ai[3] * bj[3] +
                                ai[4] * bj[4] +
                                ai[5] * bj[5] +
                                ai[6] * bj[6] +
                                ai[7] * bj[7];
            }
        } 
        std::memcpy(c->data + i*c->s0 + bc0, acc, sizeof(float) * 8); 
    }
}


void MMfloat(Mfloat *a, Mfloat *b, Mfloat *c)
{
    // Partition on multiple independent tasks
    for (int j=0; j<b->d1; j+=8) {  // columns of b
        tp.add_task(new MMfloatJob(a, 0, a->d0, b, j, j+8, c));
    }
    tp.wait_tasks_complete();
}