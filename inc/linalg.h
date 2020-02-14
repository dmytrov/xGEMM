#ifndef LINALG_H
#define LINALG_H

#include "strided_array.h"
#include "thread_pool.h"

typedef StridedArray<float> Mfloat;

class MMfloatKernelJob : Job
{
public:
    Mfloat *a;
    int ar0, ar1;
    Mfloat *b;
    int bc0, bc1;
    Mfloat *c;
    
    void execute() override;
};


void MMfloatKernelJob::execute()
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


Mfloat MMfloat(Mfloat a, Mfloat b)
{

}


#endif // LINALG_H
