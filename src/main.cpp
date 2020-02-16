
#include <iostream>
#include <boost/timer/timer.hpp>
#include "strided_array.h"
#include "linalg.h"

template<class T>
double test()
{
    int n = 1024;
    int m = 1;
    StridedArray<T> a(n, n, T(0.0));
    for (int i=0; i<n; i++)
        a.setitem(i, i, i);
    
    StridedArray<T> b(n, n, T(0.0));
    for (int i=0; i<n; i++)
        b.setitem(i, i, 1);
    
    StridedArray<T> c(n, n, T(-1.0));
    
    boost::timer::cpu_timer t;
    t.start();
    for (int i=0; i<m; i++)
       MM<T>(&a, &b, &c);
    t.stop();

    double dt = double(t.elapsed().wall) / 1.0e9;
    double flops = 2 * m * pow(n, 3) / dt;
    return flops;
}

int main(void)
{
    printf("int:    %f GFLOPS\n", 1.0e-9 * test<int>());
    printf("float:  %f GFLOPS\n", 1.0e-9 * test<float>());
    printf("double: %f GFLOPS\n", 1.0e-9 * test<double>());
    return 0;
}