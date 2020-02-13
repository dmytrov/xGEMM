#include "thread_pool.h"
#include <iostream>

//using namespace std;

int main(void)
{
    ThreadPool tp;

    for (int i=0; i<=100; i++) {
        tp.add_task(new PrintJob(i));
    }
    tp.wait_workers_exit();
    return 0;
}