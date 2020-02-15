#ifndef LINALG_H
#define LINALG_H

#include "strided_array.h"
#include "thread_pool.h"

#define NLINES 16
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
        if (bc1-bc0 != NLINES)
            throw std::invalid_argument("Number of columns must be NLINES");
        
        T acc[NLINES] __attribute__((aligned(32))); // accumulator. Aligned    
        
        for (int i=ar0; i<ar1; i++) {  // rows of a
            T *ai = a->data + i*a->s0;  // a[i, :] row
            std::memset(acc, 0, NLINES*sizeof(T));
            
            for (int j=bc0; j<bc1; j++) {  // columns of b
                T *bj = b->data + j*b->s0;

                for (int k=0; k<a->d1; k+=16) {  // items
                    T *pa = ai+k;
                    T *pb = bj+k;
                    acc[j-bc0] +=   pa[0] * pb[0] +
                                    pa[1] * pb[1] +
                                    pa[2] * pb[2] +
                                    pa[3] * pb[3] +
                                    pa[4] * pb[4] +
                                    pa[5] * pb[5] +
                                    pa[6] * pb[6] +
                                    pa[7] * pb[7] + 
                                    pa[8] * pb[8] +
                                    pa[9] * pb[9] +
                                    pa[10] * pb[10] +
                                    pa[11] * pb[11] +
                                    pa[12] * pb[12] +
                                    pa[13] * pb[13] +
                                    pa[14] * pb[14] +
                                    pa[15] * pb[15];
                }
            } 
            std::memcpy(c->data + i*c->s0 + bc0, acc, sizeof(T) * NLINES); 
        }
    }
};

template<class T>
void MM(StridedArray<T> *a, StridedArray<T> *b, StridedArray<T> *c)
{
    // Partition on multiple independent tasks
    for (int j=0; j<b->d1; j+=NLINES) {  // columns of b
        tp.add_task(new MMJob<T>(a, 0, a->d0, b, j, j+NLINES, c));
    }
    tp.wait_tasks_complete();
}

#endif // LINALG_H
