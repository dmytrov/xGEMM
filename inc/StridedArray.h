#ifndef STRIDED_ARRAY_H
#define STRIDED_ARRAY_H

#include <iostream>
#include <cstring>
#include <math.h>
#include <boost/align/aligned_alloc.hpp>

#define AVX256_ALIGNMENT 32
#define AVX512_ALIGNMENT 64
#define ALIGNMENT AVX256_ALIGNMENT

template<class T>
struct StridedArray {
    // Access as data[i*s0 + j*s1]
    //
    //
    /* data */
    int d0, d1;  // dimensions
    int s0, s1;  // strides, s0 >= d1, number of floats
    T* data;  // actual array data

    StridedArray(/* args */);
    StridedArray(T* data, int d0, int d1);
    T getitem(int i, int j) {
        if (i >= d0 || j >= d1)
            throw std::out_of_range("Index is out of range.");
        return data[i * s0 + j * s1];
    }
    ~StridedArray();
};

template<class T>
StridedArray<T>::StridedArray(/* args */) {
    d0 = 0;
    d1 = 0;
    s0 = 0;
    s1 = 1;
    data = nullptr;
}

template<class T>
StridedArray<T>::StridedArray(T* _data, int _d0, int _d1) {
    d0 = _d0;  // data dimension 0
    d1 = _d1;  // data dimension 1
    data = nullptr;
    
    int nalignedchunks = (d1 * sizeof(T) + ALIGNMENT - 1) / (ALIGNMENT);  // number of alignment chunks
    s0 = nalignedchunks * ALIGNMENT / sizeof(T);  // number of T elements
    s1 = 1;  // number of T elements
    
    // Fill aligned memory array
    data = (T*)boost::alignment::aligned_alloc(ALIGNMENT, sizeof(T) * s0 * d1);
    for (int i=0; i<d0; i++)
        std::memcpy(data+i*s0, _data+i*d1, sizeof(T)*d1);
}

template<class T>
StridedArray<T>::~StridedArray() {
    if (this->data) {
        boost::alignment::aligned_free(this->data);
    }
}


#endif // STRIDED_ARRAY_H
