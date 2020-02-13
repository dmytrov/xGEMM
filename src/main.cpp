
#include "StridedArray.h"
#include <iostream>

//using namespace std;

int main(void)
{
    float array[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0};
    StridedArray<float> x(array, 3, 4);
    
    for (int i=0; i<x.d0; i++)
    {
        for (int j=0; j<x.d1; j++)
        {
            std::cout << " " << x.getitem(i, j);
        }
        std::cout << std::endl;
    }
    return 0;
}