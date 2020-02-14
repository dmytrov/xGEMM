#include "thread_pool.h"
#include <iostream>

//using namespace std;

int main(void)
{
    ThreadPool tp;

    for (int i=0; i<100; i++)
        tp.add_task(new EmptyJob());
    
    for (int i=0; i<10; i++)
        tp.add_task(new PrintJob(i));
    tp.wait_tasks_complete();
    cout << "---" << endl;
    
    for (int i=0; i<10; i++)
        tp.add_task(new PrintJob(i));
    tp.wait_tasks_complete();

    tp.wait_workers_exit();
    return 0;
}