
#include <iostream>
#include "strided_array.h"
#include "linalg.h"


int main(void)
{
    int n = 1024;
    Mfloat a(n, n, 1);
    Mfloat b(n, n, 1);
    Mfloat c = Mfloat(a.d0, b.d1);
    
    MMfloat(&a, &b, &c);
    
    return 0;
}