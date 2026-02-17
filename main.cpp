#include <stdio.h>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include "progress.h"

int main()
{
    std::cout << "Iniciando programa" << std::flush;
    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "\r\033[2K" << std::flush;

    progress();

    std::cout << "\rModelo carregado com sucesso!" << std::endl;

    return 0;
}