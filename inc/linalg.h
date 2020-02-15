#ifndef LINALG_H
#define LINALG_H

#include "strided_array.h"
#include "thread_pool.h"

#define NLINES 16
extern ThreadPool tp;
typedef StridedArray<float> Mfloat;
typedef StridedArray<double> Mdouble;

class MMfloatJob : public Job
{
    // c = dot(a, b)
    //
    //

public:
    Mfloat *a;
    int ar0, ar1;
    Mfloat *b;
    int bc0, bc1;
    Mfloat *c;
    
    MMfloatJob(Mfloat *_a, int _ar0, int _ar1, Mfloat *_b, int _bc0, int _bc1, Mfloat *_c) {
        a = _a;
        ar0 = _ar0;
        ar1 = _ar1;
        b = _b;
        bc0 = _bc0;
        bc1 = _bc1;
        c = _c;
    }
    void execute() override;
    void execute_unoptimized();
    void execute_optimized();
};


void MMfloat(Mfloat *a, Mfloat *b, Mfloat *c);



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
            throw std::invalid_argument("Number of columns must be 8");

        for (int i=ar0; i<ar1; i++) {  // rows of a
            T *ai = a->data + i*a->s0;  // a[i, :] row
            
            T acc[NLINES] __attribute__((aligned(32))); // accumulator. Aligned
            std::memset(acc, 0, NLINES*sizeof(T));
            
            for (int j=bc0; j<bc1; j++) {  // columns of b
                T *bj = b->data + j*b->s0;

                for (int k=0; k<a->d1; k+=16) {  // items
                    T *pa = ai+k;
                    T *pb = bj+k;
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
