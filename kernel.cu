#include <iostream>           // Подключение библиотеки для ввода/вывода в консоль
#include <vector>             // Подключение библиотеки для работы с динамическими массивами std::vector
#include <algorithm>          // Подключение стандартных алгоритмов сортировки (sort, stable_sort, make_heap)
#include <chrono>             // Для измерения времени выполнения кода
#include <cstdlib>            // Для rand()
#include <climits>            // Для INT_MAX

#include <cuda_runtime.h>     // Основные функции CUDA runtime API
#include <thrust/device_vector.h>  // Thrust: упрощенная работа с GPU-векторами
#include <thrust/sort.h>           // Thrust: функция параллельной сортировки на GPU

#define BLOCK_SIZE 256        // Размер блока потоков для GPU-битонической сортировки

void cpu_merge_sort(std::vector<int>& a) {
    std::stable_sort(a.begin(), a.end());  // Сортировка слиянием на CPU (стабильная)
}
void cpu_quick_sort(std::vector<int>& a) {
    std::sort(a.begin(), a.end());         // Быстрая сортировка на CPU
}
void cpu_heap_sort(std::vector<int>& a) {
    std::make_heap(a.begin(), a.end());    // Создание бинарной кучи
    std::sort_heap(a.begin(), a.end());    // Сортировка с использованием кучи
}


__global__ void bitonic_block_sort(int* data, int n) {
    __shared__ int s[BLOCK_SIZE];          // Разделяемая память для ускорения сортировки внутри блока
    int tid = threadIdx.x;                 // Локальный индекс потока внутри блока
    int gid = blockIdx.x * blockDim.x + tid;  // Глобальный индекс потока по всему массиву

    if (gid < n) s[tid] = data[gid];      // Копируем данные в shared memory
    else s[tid] = INT_MAX;                 // Если поток вне массива, ставим максимальное значение

    __syncthreads();                       // Синхронизация потоков блока

    // Bitonic sort на shared memory
    for (int k = 2; k <= blockDim.x; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;             // XOR индекс для сравнения
            if (ixj > tid) {
                if ((tid & k) == 0) {
                    if (s[tid] > s[ixj]) { // Сравнение и обмен для восходящей последовательности
                        int tmp = s[tid]; s[tid] = s[ixj]; s[ixj] = tmp;
                    }
                }
                else {
                    if (s[tid] < s[ixj]) { // Сравнение и обмен для нисходящей последовательности
                        int tmp = s[tid]; s[tid] = s[ixj]; s[ixj] = tmp;
                    }
                }
            }
            __syncthreads();               // Синхронизация после каждой стадии
        }
    }

    if (gid < n) data[gid] = s[tid];       // Копируем отсортированный блок обратно в глобальную память
}


__global__ void merge_pair_kernel(int* src, int* dst, int n, int runSize) {
    int pairId = blockIdx.x;                         // Номер пары блоков для слияния
    int leftStart = pairId * (runSize * 2);          // Начало левой части
    int mid = leftStart + runSize;                   // Граница между левым и правым блоком
    int rightStart = mid;
    int leftEnd = min(mid, n);                       // Не выйти за массив
    int rightEnd = min(rightStart + runSize, n);

    if (threadIdx.x == 0) {                          // Только один поток блока выполняет последовательное слияние
        int i = leftStart;
        int l = leftStart;
        int r = rightStart;
        while (l < leftEnd && r < rightEnd) {
            if (src[l] <= src[r]) dst[i++] = src[l++];
            else dst[i++] = src[r++];
        }
        while (l < leftEnd) dst[i++] = src[l++];
        while (r < rightEnd) dst[i++] = src[r++];
    }
}


float gpu_merge_sort(std::vector<int>& a) {
    int n = static_cast<int>(a.size());
    if (n == 0) return 0.0f;

    int* d_src = nullptr;
    int* d_tmp = nullptr;
    size_t bytes = n * sizeof(int);

    cudaMalloc(&d_src, bytes);                     // Выделяем память на GPU под исходный массив
    cudaMalloc(&d_tmp, bytes);                     // Выделяем временный буфер

    cudaMemcpy(d_src, a.data(), bytes, cudaMemcpyHostToDevice); // Копируем данные с CPU на GPU

    auto t_start = std::chrono::high_resolution_clock::now();   // Засекаем время

    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bitonic_block_sort << <blocks, BLOCK_SIZE >> > (d_src, n);      // Сортировка блоков
    cudaDeviceSynchronize();

    int runSize = BLOCK_SIZE;
    int* d_read = d_src;
    int* d_write = d_tmp;

    while (runSize < n) {                                       // Итеративное слияние блоков
        int pairs = (n + (runSize * 2) - 1) / (runSize * 2);
        merge_pair_kernel << <pairs, BLOCK_SIZE >> > (d_read, d_write, n, runSize);
        cudaDeviceSynchronize();

        int* t = d_read; d_read = d_write; d_write = t;        // Меняем буферы
        runSize *= 2;
    }

    if (d_read != d_src) cudaMemcpy(d_src, d_read, bytes, cudaMemcpyDeviceToDevice); // Копируем обратно

    cudaMemcpy(a.data(), d_src, bytes, cudaMemcpyDeviceToHost); // Копируем результат на CPU

    cudaFree(d_src); cudaFree(d_tmp);                            // Освобождаем память

    auto t_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(t_end - t_start).count(); // Возвращаем время выполнения
}


float gpu_quick_sort(std::vector<int>& a) {
    auto t_start = std::chrono::high_resolution_clock::now();

    thrust::device_vector<int> d_vec = a;           // Копируем массив на GPU
    thrust::sort(d_vec.begin(), d_vec.end());       // Параллельная сортировка на GPU
    thrust::copy(d_vec.begin(), d_vec.end(), a.begin()); // Копируем результат обратно на CPU

    auto t_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(t_end - t_start).count();
}

float gpu_heap_sort(std::vector<int>& a) {
    // Thrust::sort используется для имитации GPU-версии HeapSort,
    // так как классическая пирамидальная сортировка плохо параллелится
    // и неэффективна для реализации на CUDA
    auto t_start = std::chrono::high_resolution_clock::now();
    thrust::device_vector<int> d_vec = a;
    thrust::sort(d_vec.begin(), d_vec.end());
    thrust::copy(d_vec.begin(), d_vec.end(), a.begin());
    auto t_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(t_end - t_start).count();
}


void benchmark(int n) {
    std::vector<int> base(n);
    for (int i = 0; i < n; ++i) base[i] = rand();    // Заполняем массив случайными числами

    std::cout << "\nРазмер массива: " << n << "\n";

    for (auto algo : { "Merge", "Quick", "Heap" }) {
        auto a = base;

        // CPU
        auto t1 = std::chrono::high_resolution_clock::now();
        if (algo == std::string("Merge")) cpu_merge_sort(a);
        if (algo == std::string("Quick")) cpu_quick_sort(a);
        if (algo == std::string("Heap"))  cpu_heap_sort(a);
        auto t2 = std::chrono::high_resolution_clock::now();
        float cpu_time = std::chrono::duration<float, std::milli>(t2 - t1).count();

        // GPU
        a = base;
        float gpu_time = 0.0f;
        if (algo == std::string("Merge")) gpu_time = gpu_merge_sort(a);
        if (algo == std::string("Quick")) gpu_time = gpu_quick_sort(a);
        if (algo == std::string("Heap"))  gpu_time = gpu_heap_sort(a);

        std::cout << algo
            << " | CPU: " << cpu_time << " ms"
            << " | GPU: " << gpu_time << " ms\n";
    }
}


int main() {
    int deviceCount = 0;
    cudaError_t st = cudaGetDeviceCount(&deviceCount); // Проверка наличия CUDA-устройств
    if (st != cudaSuccess || deviceCount == 0) {
        std::cerr << "No CUDA device available or driver mismatch: " << cudaGetErrorString(st) << "\n";
        std::cerr << "GPU-измерения будут пропущены или выполняться на CPU через Thrust (если поддерживается).\n";
    }
    else {
        std::cout << "Detected " << deviceCount << " CUDA device(s)\n";
    }

    // Бенчмарк на разных размерах массивов
    benchmark(10000);      // 10 тысяч элементов
    benchmark(100000);     // 100 тысяч элементов
    benchmark(1000000);    // 1 миллион элементов

    return 0;
}
