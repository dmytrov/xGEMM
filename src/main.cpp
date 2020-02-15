
#include <iostream>
#include <boost/timer/timer.hpp>
#include "strided_array.h"
#include "linalg.h"


int main(void)
{
    int n = 200*16;
    int m = 3;
    Mfloat a(n, n, 1);
    Mfloat b(n, n, 1);
    Mfloat c = Mfloat(a.d0, b.d1);
    
    boost::timer::cpu_timer t;
    t.start();
    for (int i=0; i<m; i++)
       MMfloat(&a, &b, &c);
    t.stop();
    double dt = double(t.elapsed().wall) / 1.0e9;
    double flops = 2 * m * pow(n, 3) / dt;
    cout << (1.0e-9 * flops) << " GFLOPS" << endl;
    return 0;
}