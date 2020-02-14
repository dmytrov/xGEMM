#ifndef LINALG_H
#define LINALG_H

#include "strided_array.h"
#include "thread_pool.h"

typedef StridedArray<float> Mfloat;

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


#endif // LINALG_H
