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

    StridedArray(int d0, int d1, T fillvalue=0);
    StridedArray(int d0, int d1, T* _data);
    T getitem(int i, int j) {
        if (i >= d0 || j >= d1)
            throw std::out_of_range("Index is out of range.");
        return data[i * s0 + j * s1];
    }
    void setitem(int i, int j, T value) {
        if (i >= d0 || j >= d1)
            throw std::out_of_range("Index is out of range.");
        data[i * s0 + j * s1] = value;
    }
    ~StridedArray();

    void print();

private:
    void init(int d0, int d1);
};

template<class T>
void StridedArray<T>::init(int _d0, int _d1) {
    data = nullptr;
    d0 = _d0;
    d1 = _d1;
    
    int nalignedchunks = (d1 * sizeof(T) + ALIGNMENT - 1) / (ALIGNMENT);  // number of alignment chunks
    s0 = nalignedchunks * ALIGNMENT / sizeof(T);  // number of T elements
    s1 = 1;  // number of T elements
    
    // Allocate aligned memory array
    data = (T*)boost::alignment::aligned_alloc(ALIGNMENT, sizeof(T) * s0 * d1);
}

template<class T>
StridedArray<T>::StridedArray(int _d0, int _d1, T fillvalue) {
    init(_d0, _d1);
    for (int j=0; j<d0 * s0; j++)
        data[j] = fillvalue;
}

template<class T>
StridedArray<T>::StridedArray(int _d0, int _d1, T* _data) {
    init(_d0, _d1);
    for (int i=0; i<d0; i++)
        std::memcpy(data+i*s0, _data+i*d1, sizeof(T)*d1);
}

template<class T>
StridedArray<T>::~StridedArray() {
    if (this->data) {
        boost::alignment::aligned_free(this->data);
    }
}

template<class T>
void StridedArray<T>::print() {
    std::cout << "[";
    for (int i=0; i<d0; i++) {
        for (int j=0; j<d1; j++) {
            std::cout << " " << getitem(i, j);
        }
        std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
}


#endif // STRIDED_ARRAY_H
