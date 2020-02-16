#ifndef LINALG_H
#define LINALG_H

#include <immintrin.h>
#include "strided_array.h"
#include "thread_pool.h"

// AVX has 32 registers. 
// Each 256-bit register can hold 8 32-bit single or 4 64-fit double floats. 
// This gives us 256 of 32-bit single floats.
// Cache line is 64 bytes = 512 bits = 2 AVX registers.
// Optimal block size is Nx16 of single floats.
// 8x16 floats fit in 16 AVX registers to store the result.

#define ALG 4

#if ALG == 1
  #define BLOCKSIZE_A 8
  #define BLOCKSIZE_B 8
#elif ALG == 2
  #define BLOCKSIZE_A 2
  #define BLOCKSIZE_B 512
#elif ALG == 3
  #define BLOCKSIZE_A 2
  #define BLOCKSIZE_B 512
#elif ALG == 4
  #define BLOCKSIZE_A 8
  #define BLOCKSIZE_B 16
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
        #elif ALG == 4
            execute4();
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

    void execute4() {
        if (bc1-bc0 != 16) 
            throw std::invalid_argument("Number of columns must be 16");
        if (ar1-ar0 != 8) 
            throw std::invalid_argument("Number of rows must be 8");
        
        // Load C in AVX
        int astride = a->s0;
        int cstride = c->s0;
        T *__restrict__ pc = c->data + ar0*cstride + bc0;
        __m256 regC0 = _mm256_load_ps(pc + 0*cstride);
        __m256 regC1 = _mm256_load_ps(pc + 0*cstride);
        __m256 regC2 = _mm256_load_ps(pc + 0*cstride);
        __m256 regC3 = _mm256_load_ps(pc + 0*cstride);
        __m256 regC4 = _mm256_load_ps(pc + 0*cstride);
        __m256 regC5 = _mm256_load_ps(pc + 0*cstride);
        __m256 regC6 = _mm256_load_ps(pc + 0*cstride);
        __m256 regC7 = _mm256_load_ps(pc + 0*cstride);
        T *__restrict__ pd = c->data + ar0*cstride + bc0 + 8;
        __m256 regD0 = _mm256_load_ps(pd + 0*cstride);
        __m256 regD1 = _mm256_load_ps(pd + 1*cstride);
        __m256 regD2 = _mm256_load_ps(pd + 2*cstride);
        __m256 regD3 = _mm256_load_ps(pd + 3*cstride);
        __m256 regD4 = _mm256_load_ps(pd + 4*cstride);
        __m256 regD5 = _mm256_load_ps(pd + 5*cstride);
        __m256 regD6 = _mm256_load_ps(pd + 6*cstride);
        __m256 regD7 = _mm256_load_ps(pd + 7*cstride);

        T *ai0 = a->data + ar0*a->s0 + 0*astride;  // a[i, :] row
        T *ai1 = a->data + ar0*a->s0 + 1*astride;
        T *ai2 = a->data + ar0*a->s0 + 2*astride;
        T *ai3 = a->data + ar0*a->s0 + 3*astride;
        T *ai4 = a->data + ar0*a->s0 + 4*astride;
        T *ai5 = a->data + ar0*a->s0 + 5*astride;
        T *ai6 = a->data + ar0*a->s0 + 6*astride;
        T *ai7 = a->data + ar0*a->s0 + 7*astride;

        for (int k=0; k < a->d1; k++) {  // full dimension walk
            T *__restrict__ bj = b->data + k*b->s0 + bc0;  // 
            
            // Left half of C. 8 of C<-A*B
            __m256 regBL = _mm256_load_ps(bj);
            __m256 regBR = _mm256_load_ps(bj+8);
            __m256 regA;
            regA = _mm256_broadcast_ss(ai0 + k);
            regC0 = _mm256_add_ps(regC0, _mm256_mul_ps(regA, regBL));
            regD0 = _mm256_add_ps(regD0, _mm256_mul_ps(regA, regBR));
            regA = _mm256_broadcast_ss(ai1 + k);
            regC1 = _mm256_add_ps(regC1, _mm256_mul_ps(regA, regBL));
            regD1 = _mm256_add_ps(regD1, _mm256_mul_ps(regA, regBR));
            regA = _mm256_broadcast_ss(ai2 + k);
            regC2 = _mm256_add_ps(regC2, _mm256_mul_ps(regA, regBL));
            regD2 = _mm256_add_ps(regD2, _mm256_mul_ps(regA, regBR));
            regA = _mm256_broadcast_ss(ai3 + k);
            regC3 = _mm256_add_ps(regC3, _mm256_mul_ps(regA, regBL));
            regD3 = _mm256_add_ps(regD3, _mm256_mul_ps(regA, regBR));
            regA = _mm256_broadcast_ss(ai4 + k);
            regC4 = _mm256_add_ps(regC4, _mm256_mul_ps(regA, regBL));
            regD4 = _mm256_add_ps(regD4, _mm256_mul_ps(regA, regBR));
            regA = _mm256_broadcast_ss(ai5 + k);
            regC5 = _mm256_add_ps(regC5, _mm256_mul_ps(regA, regBL));
            regD5 = _mm256_add_ps(regD5, _mm256_mul_ps(regA, regBR));
            regA = _mm256_broadcast_ss(ai6 + k);
            regC6 = _mm256_add_ps(regC6, _mm256_mul_ps(regA, regBL));
            regD6 = _mm256_add_ps(regD6, _mm256_mul_ps(regA, regBR));
            regA = _mm256_broadcast_ss(ai7 + k);
            regC7 = _mm256_add_ps(regC7, _mm256_mul_ps(regA, regBL));
            regD7 = _mm256_add_ps(regD7, _mm256_mul_ps(regA, regBR)); 
        }
        
        // Store AVX in C
        _mm256_store_ps(pc + 0*cstride, regC0);
        _mm256_store_ps(pc + 1*cstride, regC1);
        _mm256_store_ps(pc + 2*cstride, regC2);
        _mm256_store_ps(pc + 3*cstride, regC3);
        _mm256_store_ps(pc + 4*cstride, regC4);
        _mm256_store_ps(pc + 5*cstride, regC5);
        _mm256_store_ps(pc + 6*cstride, regC6);
        _mm256_store_ps(pc + 7*cstride, regC7);
        
        _mm256_store_ps(pd + 0*cstride, regD0);
        _mm256_store_ps(pd + 1*cstride, regD1);
        _mm256_store_ps(pd + 2*cstride, regD2);
        _mm256_store_ps(pd + 3*cstride, regD3);
        _mm256_store_ps(pd + 4*cstride, regD4);
        _mm256_store_ps(pd + 5*cstride, regD5);
        _mm256_store_ps(pd + 6*cstride, regD6);
        _mm256_store_ps(pd + 7*cstride, regD7); 
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
