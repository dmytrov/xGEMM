
#include <iostream>
#include "strided_array.h"
#include "linalg.h"

//using namespace std;

int main(void)
{
    float array[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    Mfloat a(3, 4, array);
    Mfloat b(4, 2, array);
    a.print();
    b.print();

    MMfloatKernelJob mmprod;
    mmprod.a = &a;
    mmprod.ar0 = 0;
    mmprod.ar1 = a.d0;
    mmprod.b = &b;
    mmprod.bc0 = 0;
    mmprod.bc1 = b.d1;

    Mfloat c = Mfloat(a.d0, b.d1);
    mmprod.c = &c;
    mmprod.execute();
    c.print();
    
    return 0;
}