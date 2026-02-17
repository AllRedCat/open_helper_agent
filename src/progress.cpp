#include <iostream>
#include <thread>
#include <chrono>
#include "progress.h"

int progress()
{
    for (int i = 0; i <= 100; i += 10)
    {
        std::cout << "\r\033[2KProgresso: " << i << "%" << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    std::cout << std::endl;

    return 0;
}