
#include <iostream>
#include <boost/timer/timer.hpp>
#include "strided_array.h"
#include "linalg.h"

template<class T>
void test()
{
    int n = 100*16;
    int m = 10;
    StridedArray<T> a(n, n, T(2.2));
    StridedArray<T> b(n, n, T(3.3));
    StridedArray<T> c(n, n, T(0.0));
    
    boost::timer::cpu_timer t;
    t.start();
    for (int i=0; i<m; i++)
       MM<T>(&a, &b, &c);
    t.stop();
    
    double dt = double(t.elapsed().wall) / 1.0e9;
    double flops = 2 * m * pow(n, 3) / dt;
    cout << (1.0e-9 * flops) << " GFLOPS" << endl;
}

int main(void)
{
    test<int>();
    test<float>();
    test<double>();
    return 0;
}