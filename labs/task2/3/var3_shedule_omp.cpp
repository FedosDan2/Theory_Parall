#include <iostream>
#include <omp.h>
#include <cstdio>
#include <vector>
#include <inttypes.h> 
#include<cmath>
#include <string>

#define EPS 1e-5
#define MAX_ITER 10000
#define N 1000

using TYPE = std::vector<double>;

void init(TYPE& A, TYPE& b) 
{
    A.resize(N * N);
    b.resize(N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++){
            if(i == j) 
                A[i * N + j] = 2.0;
            else 
                A[i * N + j] = 1.0;
        }
    } 

    for (int j = 0; j < N; j++)
        b[j] = static_cast<double>(N+1);
}

//Вариант 3: Провести исследование на определение оптимальных параметров #pragma omp for schedule(...) при некотором фиксированном размере задачи и количестве потоков.
void lin_equation_omp3(const TYPE& A, TYPE& x, const TYPE& b, int num_threads, std::string type, int chunk_size)
{   
    // прочитать про schedule static dynamic guided
    bool converged = false;
    double tau = 0.01;
    #pragma omp parallel num_threads(num_threads)
    {   
        TYPE Ax(N);
        for (int iter = 0; iter < MAX_ITER && !converged; iter++) {
            std::fill(Ax.begin(), Ax.end(), 0.0);
            // === Этап 1: Вычисление Ax ===

            // разделяет данные между потоками статично, размеров chunk, по очереди
            // chunks - блок итераций которые выдаются на поток
            // Каждый поток получает несколько чанков
            //  чем больше chunk, тем меньше оверхед на распределение, но выше риск дисбаланса нагрузки
            if (type == "static") { 
                #pragma omp for schedule(static, chunk_size)
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++)
                        Ax[i] += A[i * N + j] * x[j];
                }
            } 
            
            else if (type == "dynamic") {
                #pragma omp for schedule(dynamic, chunk_size)
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++)
                        Ax[i] += A[i * N + j] * x[j];
                }
            } else if (type == "guided") {
                #pragma omp for schedule(guided, chunk_size)
                for (int i = 0; i < N; i++) {
                    for (int j = 0; j < N; j++)
                        Ax[i] += A[i * N + j] * x[j];
                }
            }
            #pragma omp barrier


            if (type == "static") {
                #pragma omp for schedule(static, chunk_size)
                for (int i = 0; i < N; i++)
                    x[i] = x[i] - tau * (Ax[i] - b[i]);
            } else if (type == "dynamic") {
                #pragma omp for schedule(dynamic, chunk_size)
                for (int i = 0; i < N; i++)
                    x[i] = x[i] - tau * (Ax[i] - b[i]);
            } else if (type == "guided") {
                #pragma omp for schedule(guided, chunk_size)
                for (int i = 0; i < N; i++)
                    x[i] = x[i] - tau * (Ax[i] - b[i]);
            }
            #pragma omp barrier
    

            double global_diff_norm = 0.0;
            if (type == "static") {
                #pragma omp for schedule(static, chunk_size) reduction(+:global_diff_norm)
                for (int i = 0; i < N; i++) {
                    double diff = tau * (Ax[i] - b[i]);
                    global_diff_norm += diff * diff;
                }
            } else if (type == "dynamic") {
                #pragma omp for schedule(dynamic, chunk_size) reduction(+:global_diff_norm)
                for (int i = 0; i < N; i++) {
                    double diff = tau * (Ax[i] - b[i]);
                    global_diff_norm += diff * diff;
                }
            } else if (type == "guided") {
                #pragma omp for schedule(guided, chunk_size) reduction(+:global_diff_norm)
                for (int i = 0; i < N; i++) {
                    double diff = tau * (Ax[i] - b[i]);
                    global_diff_norm += diff * diff;
                }
            }
            
            #pragma omp single
            {
                if (sqrt(global_diff_norm) < EPS) {
                    converged = true;
                }
            }

            #pragma omp barrier 

            if (converged) {
                break;
            }
        }  
    }
}



double run_parallel(const TYPE& A, TYPE& x, const TYPE& b, int num_threads, std::string type, int chunk_size) 
{   
    double t = omp_get_wtime();
    lin_equation_omp3(A, x, b, num_threads, type, chunk_size);
    t = omp_get_wtime() - t;
    return t;
}


int main() 
{
    TYPE A, b;
    TYPE x(N, 0.0);   
    
    init(A, b);
    
    const std::vector<std::string> schedules = {"static", "dynamic", "guided"};
    const std::vector<int> chunks = {1, 4, 8, 16, 32, 64, 128, 256};
    TYPE results;

    printf("Linear equation (N=%d)\n", N);
    printf("Memory: %" PRIu64 " MiB\n", static_cast<uint64_t>(N * N + 2 * N) * sizeof(double) / (1024 * 1024));

    const int runs = 20;  // Усреднение
        
    FILE* file = fopen("results/var3/results2.txt", "w");
    if (file) {
        fprintf(file, "%s %d %s\n", "Average Results after", runs, "operations on 8 threads:");
        fprintf(file, "%-10s %15s %15s\n", "Type schedule",  "Chunks", "Time");
        results.resize(schedules.size() * chunks.size());
        int i = 0;

        for(std::string schedule : schedules)
        {   
            for (int chunk : chunks){
                double total_time = 0.0;

                for (int r = 0; r < runs; r++) {
                    std::fill(x.begin(), x.end(), 0.0);  // Сброс перед каждым запуском
                    total_time += run_parallel(A, x, b, 8, schedule, chunk);
                }
                results[i] = total_time / runs;
                printf("Type: %s\t Chunks: %d\t done", schedule.c_str(), chunk);
                fprintf(file, "%-10s %15d %15.6f\n", schedule.c_str(), chunk, results[i]);
                ++i;
            }
            fprintf(file, "\n");
            
        }
        fclose(file);
    }
    return 0;
}