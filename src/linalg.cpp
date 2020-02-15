#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline") //Optimization flags
#pragma GCC option("arch=native","tune=native","no-zero-upper") //Enable AVX
#pragma GCC target("avx")  //Enable AVX

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
    if (bc1-bc0 != NLINES)
        throw std::invalid_argument("Number of columns must be 8");

    

    for (int i=ar0; i<ar1; i++) {  // rows of a
        float *ai = a->data + i*a->s0;  // a[i, :] row
        
        float acc[NLINES] __attribute__((aligned(32))); // accumulator. Aligned
        std::memset(acc, 0, NLINES*sizeof(float));
        
        for (int j=bc0; j<bc1; j++) {  // columns of b
            float *bj = b->data + j*b->s0;

            for (int k=0; k<a->d1; k+=16) {  // items
                float *pa = ai+k;
                float *pb = bj+k;
                acc[j-bc0] +=   ai[0] * bj[0] +
                                ai[1] * bj[1] +
                                ai[2] * bj[2] +
                                ai[3] * bj[3] +
                                ai[4] * bj[4] +
                                ai[5] * bj[5] +
                                ai[6] * bj[6] +
                                ai[7] * bj[7] + 
                                ai[8] * bj[8] +
                                ai[9] * bj[9] +
                                ai[10] * bj[10] +
                                ai[11] * bj[11] +
                                ai[12] * bj[12] +
                                ai[13] * bj[13] +
                                ai[14] * bj[14] +
                                ai[15] * bj[15];
            }
        } 
        std::memcpy(c->data + i*c->s0 + bc0, acc, sizeof(float) * NLINES); 
    }
}


void MMfloat(Mfloat *a, Mfloat *b, Mfloat *c)
{
    // Partition on multiple independent tasks
    for (int j=0; j<b->d1; j+=NLINES) {  // columns of b
        tp.add_task(new MMfloatJob(a, 0, a->d0, b, j, j+NLINES, c));
    }
    tp.wait_tasks_complete();
}