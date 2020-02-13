#include "thread_pool.h"
#include <iostream>

//using namespace std;

int main(void)
{
    ThreadPool tp;

    for (int i=0; i<=10000000; i++) {
        tp.add_task(i);
    }
    tp.wait_workers_exit();
    return 0;
}