#ifndef LINALG_H
#define LINALG_H

#include <immintrin.h>
#include "strided_array.h"
#include "thread_pool.h"

// AVX has 32 registers. 
// Each 256-bit register can hold 8 32-bit single or 4 64-fit double floats. 
// This gives us 256 of 32-bit single floats.
// Cache line is 64 bytes = 512 bits = 2 AVX registers = 16 single floats.
// Optimal block size is Nx16 of single floats.
// 8x16 floats fit in 16 AVX registers to store the result.

#define ALG 4 

#if ALG == 0
  #define BLOCKSIZE_A 16
  #define BLOCKSIZE_B 16
#elif ALG == 1
  #define BLOCKSIZE_A 256
  #define BLOCKSIZE_B 256
#elif ALG == 2
  #define BLOCKSIZE_A 256
  #define BLOCKSIZE_B 256
#elif ALG == 3
  #define BLOCKSIZE_A 2
  #define BLOCKSIZE_B 512
#elif ALG == 4
  #define BLOCKSIZE_A 6
  #define BLOCKSIZE_B 16
#elif ALG == 5
  #define BLOCKSIZE_A 4
  #define BLOCKSIZE_B 4
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
        #if ALG == 0
            execute0();
        #elif ALG == 1
            execute1();
        #elif ALG == 2
            execute2();
        #elif ALG == 3
            execute3();
        #elif ALG == 4
            execute4();
        #elif ALG == 5
            execute5();
        #endif
    }

    void execute0() {
        T *A[ar1-ar0];
        for (int i=0; i<ar1-ar0; i++)
            A[i] = a->data + i*a->s0;
        T *B[b->d0];
        for (int j=0; j<b->d0; j++)
            B[j] = b->data + j*b->s0 + bc0;
        T *C[ar1-ar0];
        for (int i=0; i<ar1-ar0; i++)
            C[i] = c->data + i*a->s0 + bc0;

        for (int i=0; i<ar1-ar0; i++)
            for (int j=0; j<bc1-bc0; j++)
                for (int k=0; k<a->d1; k++)
                    C[i][j] += A[i][k] * B[k][j];
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
        if (ar1-ar0 != 6) 
            throw std::invalid_argument("Number of rows must be 6");
        
        // Load C in AVX
        int astride = a->s0;
        int bstride = b->s0;
        int cstride = c->s0;

        T* astart = a->data + ar0*a->s0;
        T* bstart = b->data + bc0;
        
        T *__restrict__ pc = c->data + ar0*cstride + bc0;
        __m256 regC0 = _mm256_load_ps(pc + 0*cstride);
        __m256 regC1 = _mm256_load_ps(pc + 0*cstride);
        __m256 regC2 = _mm256_load_ps(pc + 0*cstride);
        __m256 regC3 = _mm256_load_ps(pc + 0*cstride);
        __m256 regC4 = _mm256_load_ps(pc + 0*cstride);
        __m256 regC5 = _mm256_load_ps(pc + 0*cstride);
        T *__restrict__ pd = c->data + ar0*cstride + bc0 + 8;
        __m256 regD0 = _mm256_load_ps(pd + 0*cstride);
        __m256 regD1 = _mm256_load_ps(pd + 1*cstride);
        __m256 regD2 = _mm256_load_ps(pd + 2*cstride);
        __m256 regD3 = _mm256_load_ps(pd + 3*cstride);
        __m256 regD4 = _mm256_load_ps(pd + 4*cstride);
        __m256 regD5 = _mm256_load_ps(pd + 5*cstride);
  
        const int offset0 = astride * 0;  // a[i, :] row
        const int offset1 = astride * 1;
        const int offset2 = astride * 2;
        const int offset3 = astride * 3;
        const int offset4 = astride * 4;
        const int offset5 = astride * 5;
        
        for (int k=0; k < a->d1; k++) {  // full dimension walk
            // Left half of C. 8 of C<-A*B
            __m256 regBL = _mm256_load_ps(bstart);
            __m256 regBR = _mm256_load_ps(bstart + 8);

#define USE_PREFETCH4
#ifdef USE_PREFETCH4
            // Prefetch 
            #define FLOATSINCACHELINE4 16
            #define PREFETCHAHEAD4 8
            _mm_prefetch(bstart + bstride*PREFETCHAHEAD4, _MM_HINT_T0);
#endif
            regC0 = _mm256_add_ps(regC0, _mm256_mul_ps(_mm256_set1_ps(astart[offset0]), regBL));
            regD0 = _mm256_add_ps(regD0, _mm256_mul_ps(_mm256_set1_ps(astart[offset0]), regBR));
            regC1 = _mm256_add_ps(regC1, _mm256_mul_ps(_mm256_set1_ps(astart[offset1]), regBL));
            regD1 = _mm256_add_ps(regD1, _mm256_mul_ps(_mm256_set1_ps(astart[offset1]), regBR));
            regC2 = _mm256_add_ps(regC2, _mm256_mul_ps(_mm256_set1_ps(astart[offset2]), regBL));
            regD2 = _mm256_add_ps(regD2, _mm256_mul_ps(_mm256_set1_ps(astart[offset2]), regBR));
            regC3 = _mm256_add_ps(regC3, _mm256_mul_ps(_mm256_set1_ps(astart[offset3]), regBL));
            regD3 = _mm256_add_ps(regD3, _mm256_mul_ps(_mm256_set1_ps(astart[offset3]), regBR));
            regC4 = _mm256_add_ps(regC4, _mm256_mul_ps(_mm256_set1_ps(astart[offset4]), regBL));
            regD4 = _mm256_add_ps(regD4, _mm256_mul_ps(_mm256_set1_ps(astart[offset4]), regBR));
            regC5 = _mm256_add_ps(regC5, _mm256_mul_ps(_mm256_set1_ps(astart[offset5]), regBL));
            regD5 = _mm256_add_ps(regD5, _mm256_mul_ps(_mm256_set1_ps(astart[offset5]), regBR));
            astart++;
            bstart += bstride;
        }
        
        // Store AVX in C, left
        _mm256_store_ps(pc + 0*cstride, regC0);
        _mm256_store_ps(pc + 1*cstride, regC1);
        _mm256_store_ps(pc + 2*cstride, regC2);
        _mm256_store_ps(pc + 3*cstride, regC3);
        _mm256_store_ps(pc + 4*cstride, regC4);
        _mm256_store_ps(pc + 5*cstride, regC5);
        // Store AVX in C, right
        _mm256_store_ps(pd + 0*cstride, regD0);
        _mm256_store_ps(pd + 1*cstride, regD1);
        _mm256_store_ps(pd + 2*cstride, regD2);
        _mm256_store_ps(pd + 3*cstride, regD3);
        _mm256_store_ps(pd + 4*cstride, regD4);
        _mm256_store_ps(pd + 5*cstride, regD5);
    }

    
    void execute5() {
        if (bc1-bc0 != 4) 
            throw std::invalid_argument("Number of columns must be 4");
        if (ar1-ar0 != 4) 
            throw std::invalid_argument("Number of rows must be 4");
        
        // Load C in AVX
        int astride = a->s0;
        int bstride = b->s0;
        int cstride = c->s0;

        __m256 regC00 = _mm256_set1_ps(0);
        __m256 regC01 = _mm256_set1_ps(0);
        __m256 regC02 = _mm256_set1_ps(0);
        __m256 regC03 = _mm256_set1_ps(0);
        __m256 regC10 = _mm256_set1_ps(0);
        __m256 regC11 = _mm256_set1_ps(0);
        __m256 regC12 = _mm256_set1_ps(0);
        __m256 regC13 = _mm256_set1_ps(0);
        __m256 regC20 = _mm256_set1_ps(0);
        __m256 regC21 = _mm256_set1_ps(0);
        __m256 regC22 = _mm256_set1_ps(0);
        __m256 regC23 = _mm256_set1_ps(0);
        __m256 regC30 = _mm256_set1_ps(0);
        __m256 regC31 = _mm256_set1_ps(0);
        __m256 regC32 = _mm256_set1_ps(0);
        __m256 regC33 = _mm256_set1_ps(0);
        
        T *ai0 = a->data + ar0*a->s0 + 0*astride;  // a[i, :] row
        T *ai1 = a->data + ar0*a->s0 + 1*astride;  // a[i, :] row
        T *ai2 = a->data + ar0*a->s0 + 2*astride;  // a[i, :] row
        T *ai3 = a->data + ar0*a->s0 + 3*astride;  // a[i, :] row

        T *bj0 = b->data + bc0*b->s0 + 0*bstride;  // b[:, j] column
        T *bj1 = b->data + bc0*b->s0 + 1*bstride;  // b[:, j] column
        T *bj2 = b->data + bc0*b->s0 + 2*bstride;  // b[:, j] column
        T *bj3 = b->data + bc0*b->s0 + 3*bstride;  // b[:, j] column
        
        
        for (int k=0; k < a->d1; k+=8) {  // full dimension walk
            
//#define USE_PREFETCH5
#ifdef USE_PREFETCH5
            // Prefetch 
            #define PREFETCHAHEAD5 16
            _mm_prefetch(ai0 + k + PREFETCHAHEAD5, _MM_HINT_T0);
            _mm_prefetch(ai1 + k + PREFETCHAHEAD5, _MM_HINT_T0);
            _mm_prefetch(ai2 + k + PREFETCHAHEAD5, _MM_HINT_T0);
            _mm_prefetch(ai3 + k + PREFETCHAHEAD5, _MM_HINT_T0);

            _mm_prefetch(bj0 + k + PREFETCHAHEAD5, _MM_HINT_T0);
            _mm_prefetch(bj1 + k + PREFETCHAHEAD5, _MM_HINT_T0);
            _mm_prefetch(bj2 + k + PREFETCHAHEAD5, _MM_HINT_T0);
            _mm_prefetch(bj3 + k + PREFETCHAHEAD5, _MM_HINT_T0);
#endif
            __m256 regA0 = _mm256_load_ps(ai0 + k);
            __m256 regA1 = _mm256_load_ps(ai1 + k);
            __m256 regA2 = _mm256_load_ps(ai2 + k);
            __m256 regA3 = _mm256_load_ps(ai3 + k);

            __m256 regB0 = _mm256_load_ps(bj0 + k);
            __m256 regB1 = _mm256_load_ps(bj1 + k);
            __m256 regB2 = _mm256_load_ps(bj2 + k);
            __m256 regB3 = _mm256_load_ps(bj3 + k);

            regC00 = _mm256_add_ps(regC00, _mm256_mul_ps(regA0, regB0));
            regC01 = _mm256_add_ps(regC01, _mm256_mul_ps(regA0, regB1));
            regC02 = _mm256_add_ps(regC02, _mm256_mul_ps(regA0, regB2));
            regC03 = _mm256_add_ps(regC03, _mm256_mul_ps(regA0, regB3));

            regC10 = _mm256_add_ps(regC10, _mm256_mul_ps(regA1, regB0));
            regC11 = _mm256_add_ps(regC11, _mm256_mul_ps(regA1, regB1));
            regC12 = _mm256_add_ps(regC12, _mm256_mul_ps(regA1, regB2));
            regC13 = _mm256_add_ps(regC13, _mm256_mul_ps(regA1, regB3));

            regC20 = _mm256_add_ps(regC20, _mm256_mul_ps(regA2, regB0));
            regC21 = _mm256_add_ps(regC21, _mm256_mul_ps(regA2, regB1));
            regC22 = _mm256_add_ps(regC22, _mm256_mul_ps(regA2, regB2));
            regC23 = _mm256_add_ps(regC23, _mm256_mul_ps(regA2, regB3));

            regC30 = _mm256_add_ps(regC30, _mm256_mul_ps(regA3, regB0));
            regC31 = _mm256_add_ps(regC31, _mm256_mul_ps(regA3, regB1));
            regC32 = _mm256_add_ps(regC32, _mm256_mul_ps(regA3, regB2));
            regC33 = _mm256_add_ps(regC33, _mm256_mul_ps(regA3, regB3));
        }
        // Store C
        #define ARRAY8SUM(x) (x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7])
        T *__restrict__ pc = c->data + ar0*cstride + bc0;
        pc[0*cstride + 0] = ARRAY8SUM(regC00);
        pc[0*cstride + 1] = ARRAY8SUM(regC01);
        pc[0*cstride + 2] = ARRAY8SUM(regC02);
        pc[0*cstride + 3] = ARRAY8SUM(regC03);

        pc[1*cstride + 0] = ARRAY8SUM(regC10);
        pc[1*cstride + 1] = ARRAY8SUM(regC11);
        pc[1*cstride + 2] = ARRAY8SUM(regC12);
        pc[1*cstride + 3] = ARRAY8SUM(regC13);

        pc[2*cstride + 0] = ARRAY8SUM(regC20);
        pc[2*cstride + 1] = ARRAY8SUM(regC21);
        pc[2*cstride + 2] = ARRAY8SUM(regC22);
        pc[2*cstride + 3] = ARRAY8SUM(regC23);

        pc[3*cstride + 0] = ARRAY8SUM(regC30);
        pc[3*cstride + 1] = ARRAY8SUM(regC31);
        pc[3*cstride + 2] = ARRAY8SUM(regC32);
        pc[3*cstride + 3] = ARRAY8SUM(regC33);
        
    }
};

template<class T>
void MM(StridedArray<T> *a, StridedArray<T> *b, StridedArray<T> *c)
{
    // Partition on multiple independent tasks
    for (int i=0; i+BLOCKSIZE_A<=a->d0; i+=BLOCKSIZE_A)  // columns of a
        for (int j=0; j+BLOCKSIZE_B<=b->d1; j+=BLOCKSIZE_B)  // columns of b
            tp.add_task(new MMJob<T>(a, i, i+BLOCKSIZE_A, b, j, j+BLOCKSIZE_B, c));
    tp.wait_tasks_complete();
}

#endif // LINALG_H
