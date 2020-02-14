
#include <iostream>
#include "strided_array.h"
#include "linalg.h"

//using namespace std;

int main(void)
{
    float array[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    int n = 1000;
    Mfloat a(n, n);
    Mfloat b(n, n);
    Mfloat c = Mfloat(a.d0, b.d1);
    
    MMfloat(&a, &b, &c);
    c.print();
    
    return 0;
}