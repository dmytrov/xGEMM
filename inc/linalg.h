#ifndef LINALG_H
#define LINALG_H

#include "strided_array.h"
#include "thread_pool.h"

#define BLOCKSIZE 8
extern ThreadPool tp;

template<class T>
class MMJob : public Job
{
public:
    StridedArray<T> *a;
    int ar0, ar1;
    StridedArray<T> *b;
    int bc0, bc1;
    StridedArray<T> *c;
    
    MMJob(StridedArray<T> *_a, int _ar0, int _ar1, StridedArray<T> *_b, int _bc0, int _bc1, StridedArray<T> *_c) {
        a = _a;
        ar0 = _ar0;
        ar1 = _ar1;
        b = _b;
        bc0 = _bc0;
        bc1 = _bc1;
        c = _c;
    }

    void execute() override {
         // Use 8 columns to stay within chache
        if (bc1-bc0 != BLOCKSIZE)
            throw std::invalid_argument("Number of columns must be BLOCKSIZE");
        
        T acc[BLOCKSIZE] __attribute__((aligned(32))); // accumulator. Aligned    
        
        for (int i=ar0; i<ar1; i++) {  // rows of a
            T *ai = a->data + i*a->s0;  // a[i, :] row
            std::memset(acc, 0, BLOCKSIZE*sizeof(T));
            
            for (int j=bc0; j<bc1; j++) {  // columns of b
                T *bj = b->data + j*b->s0;

                T dot = 0;
                int k = 0;
                while (k < a->d1) {
                    dot +=  ai[k+0] * bj[k+0] +
                            ai[k+1] * bj[k+1] +
                            ai[k+2] * bj[k+2] +
                            ai[k+3] * bj[k+3] +
                            ai[k+4] * bj[k+4] +
                            ai[k+5] * bj[k+5] +
                            ai[k+6] * bj[k+6] +
                            ai[k+7] * bj[k+7];
                    k += 8;
                }
                acc[j-bc0] = dot;
            } 
            std::memcpy(c->data + i*c->s0 + bc0, acc, sizeof(T) * BLOCKSIZE); 
        }
    }
};

template<class T>
void MM(StridedArray<T> *a, StridedArray<T> *b, StridedArray<T> *c)
{
    // Partition on multiple independent tasks
    for (int i=0; i<a->d0; i+=BLOCKSIZE)  // columns of b
        for (int j=0; j<b->d1; j+=BLOCKSIZE)  // columns of b
            tp.add_task(new MMJob<T>(a, i, i+BLOCKSIZE, b, j, j+BLOCKSIZE, c));
    tp.wait_tasks_complete();
}

#endif // LINALG_H
