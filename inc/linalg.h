#ifndef LINALG_H
#define LINALG_H

#include <immintrin.h>
#include "strided_array.h"
#include "thread_pool.h"

#define ALG 3

#if ALG == 1
  #define BLOCKSIZE_A 8
  #define BLOCKSIZE_B 8
#elif ALG == 2
  #define BLOCKSIZE_A 2
  #define BLOCKSIZE_B 512
#elif ALG == 3
  #define BLOCKSIZE_A 2
  #define BLOCKSIZE_B 512
#endif



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
        #if ALG == 1
            execute2();
        #elif ALG == 2
            execute2();
        #elif ALG == 3
            execute3();
        #endif
    }

    void execute1() {
         // Use BLOCKSIZE columns to stay within chache
        if (bc1-bc0 != BLOCKSIZE_B)  // 8 is OK
            throw std::invalid_argument("Number of columns must be BLOCKSIZE");
        
        T acc[BLOCKSIZE_B] __attribute__((aligned(32))); // accumulator. Aligned    
        
        for (int i=ar0; i<ar1; i++) {  // rows of a
            T *__restrict__ ai = a->data + i*a->s0;  // a[i, :] row
            std::memset(acc, 0, BLOCKSIZE_B*sizeof(T));
            
            for (int j=bc0; j<bc1; j++) {  // columns of b
                T *__restrict__ bj = b->data + j*b->s0;

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
            std::memcpy(c->data + i*c->s0 + bc0, acc, sizeof(T) * BLOCKSIZE_B); 
        }
    }

    void execute2() {
         // Use BLOCKSIZE columns to stay within chache
        if (bc1-bc0 != BLOCKSIZE_B) // 128 is OK
            throw std::invalid_argument("Number of columns must be BLOCKSIZE");
        
        for (int k=0; k < a->d1; ++k) {  // full dimension walk
            T *__restrict__ ai = a->data + ar0*a->s0 + k;  // a[i, :] row
            T *__restrict__ bj = b->data + k*b->s0 + bc0;  // 
            
            for (int i=0; i<ar1-ar0; ++i) {  // rows of a
                T *__restrict__ pc = c->data + (ar0+i)*c->s0 + bc0;
                T aij = ai[i*a->s0];
                for (int j=0; j<bc1-bc0; ++j)
                    pc[j] += aij * bj[j];
            } 
        }
    }

    void execute3() {
         // Use BLOCKSIZE columns to stay within chache
        if (bc1-bc0 != BLOCKSIZE_B) // 128 is OK
            throw std::invalid_argument("Number of columns must be BLOCKSIZE");
        
        for (int k=0; k < a->d1; ++k) {  // full dimension walk
            T *__restrict__ ai = a->data + ar0*a->s0 + k;  // a[i, :] row
            T *__restrict__ bj = b->data + k*b->s0 + bc0;  // 
            
            for (int i=0; i<ar1-ar0; ++i) {  // rows of a
                T *__restrict__ pc = c->data + (ar0+i)*c->s0 + bc0;
                T * pa = ai + i*a->s0;
                __m256 regA = _mm256_broadcast_ss(pa);
                for (int j=0; j<bc1-bc0; j+=8) {
                    __m256 regB = _mm256_load_ps(bj+j);
                    __m256 regC = _mm256_load_ps(pc+j);
                    __m256 regAB = _mm256_mul_ps(regA, regB);
                    _mm256_store_ps(pc+j, _mm256_add_ps(regC, regAB));
                }
            } 
        }
    }

};

template<class T>
void MM(StridedArray<T> *a, StridedArray<T> *b, StridedArray<T> *c)
{
    // Partition on multiple independent tasks
    for (int i=0; i<a->d0; i+=BLOCKSIZE_A)  // columns of a
        for (int j=0; j<b->d1; j+=BLOCKSIZE_B)  // columns of b
            tp.add_task(new MMJob<T>(a, i, i+BLOCKSIZE_A, b, j, j+BLOCKSIZE_B, c));
    tp.wait_tasks_complete();
}

#endif // LINALG_H
