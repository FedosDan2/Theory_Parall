#include <iostream>
#include <omp.h>
#include <cstdio>
#include <vector>
#include <inttypes.h> 
#include <cstdio>
#include<cmath>
#include <string>

#define EPS 1e-5
#define MAX_ITER 10000
#define N 500

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

// Вариант 1: Для каждого распараллеливаемого цикла создается отдельная параллельная секция #pragma omp parallel for
void lin_equation_omp1(const TYPE& A, TYPE x, TYPE& b, int num_threads)
{
    double tau = 0.01;
    {
        for (int iter = 0; iter < MAX_ITER; iter++) {
            TYPE Ax(N, 0.0);
            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    Ax[i] += A[i * N + j] * x[j];
                }
            }

            #pragma omp parallel for num_threads(num_threads)
            for (int i = 0; i < N; i++) {
                x[i] = x[i] - tau * (Ax[i] - b[i]);
            }

    
            double global_diff_norm = 0.0;
            #pragma omp parallel for reduction(+:global_diff_norm) num_threads(num_threads)
            for (int i = 0; i < N; i++) {
                double diff = tau * (Ax[i] - b[i]);
                global_diff_norm += diff * diff;
            }
            
            if (sqrt(global_diff_norm) < EPS) {
                break;
            }
        }  
    }
}


// Вариант 2: Cоздается одна параллельная секция #pragma omp parallel, охватывающая весь итерационный алгоритм
void lin_equation_omp2(const TYPE& A, TYPE x, TYPE& b, int num_threads)
{
    double tau = 0.01;
    #pragma omp parallel num_threads(num_threads)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();

        int items_per_thread = N / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (N - 1) : (lb + items_per_thread - 1);

        for (int iter = 0; iter < MAX_ITER; iter++) {
            TYPE Ax(N, 0.0);
            for (int i = lb; i < ub; i++) {
                for (int j = 0; j < N; j++) {
                    Ax[i] += A[i * N + j] * x[j];
                }
            }

            #pragma omp barrier
            for (int i = lb; i < ub; i++) {
                x[i] = x[i] - tau * (Ax[i] - b[i]);
            }

            #pragma omp barrier

            double global_diff_norm = 0.0;
            #pragma omp for reduction(+:global_diff_norm)
            for (int i = 0; i < N; i++) {
                double diff = tau * (Ax[i] - b[i]);
                global_diff_norm += diff * diff;
            }
            #pragma omp barrier
            
            #pragma omp single
            if (sqrt(global_diff_norm) < EPS) {
                break;
            }

            #pragma omp barrier
        }  
    }
}

//Вариант 3: Провести исследование на определение оптимальных параметров #pragma omp for schedule(...) при некотором фиксированном размере задачи и количестве потоков.
void lin_equation_omp3(const TYPE& A, TYPE x, TYPE& b, int num_threads, std::string type, int chunk_size)
{
    bool converged = false;
    double tau = 0.01;
    #pragma omp parallel num_threads(num_threads)
    {   
        for (int iter = 0; iter < MAX_ITER && !converged; iter++) {
            TYPE Ax(N, 0.0);
            // === Этап 1: Вычисление Ax ===
            if (type == "static") {
                #pragma omp for schedule(static, chunk_size)
                for (int i = 0; i < N; i++) {
                    Ax[i] = 0.0;
                    for (int j = 0; j < N; j++)
                        Ax[i] += A[i * N + j] * x[j];
                }
            } else if (type == "dynamic") {
                #pragma omp for schedule(dynamic, chunk_size)
                for (int i = 0; i < N; i++) {
                    Ax[i] = 0.0;
                    for (int j = 0; j < N; j++)
                        Ax[i] += A[i * N + j] * x[j];
                }
            } else if (type == "guided") {
                #pragma omp for schedule(guided, chunk_size)
                for (int i = 0; i < N; i++) {
                    Ax[i] = 0.0;
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


double run_parallel(const TYPE& A, const TYPE& x, TYPE& b, int num_threads, std::string type, int chunk_size) 
{   
    double t = omp_get_wtime();
    if(type == "var1"){
        lin_equation_omp1(A, x, b, num_threads);
    }else if(type == "var2"){
        lin_equation_omp2(A, x, b, num_threads);
    }else{
        lin_equation_omp3(A, x, b, num_threads, type, chunk_size);
    }
    
    t = omp_get_wtime() - t;
    return t;
}

int main() 
{
    TYPE A, b;
    TYPE x(N, 0.0);
    init(A, b);
    
    const std::vector<std::string> schedules = {"static", "dynamic", "guided"};
    const std::vector<int> chunks = {8, 16, 32, 64, 128};
    std::vector<int> list_threads = {1, 2, 4, 6, 8, 16, 20, 40};
    TYPE results;
    
    int flag = 2; //выбор функционала
    // 1 - не shedule
    // 2 - shedule

    printf("Linear equation\n");
    printf("Memory used: %" PRIu64 " MiB\n", (N * N + N + N) * sizeof(double) / (1024 * 1024));
    if(flag ==1)
    {
        results.resize(list_threads.size());
        for (size_t i = 0; i < list_threads.size(); i++){
            std::fill(x.begin(), x.end(), 0.0);
            results[i] = run_parallel(A, x, b, list_threads[i], "var1", 0);
            printf("Threads: %2d | Time: %.6f sec\n", list_threads[i], results[i]);
        }
    } 
    else if (flag == 2) 
    {
        FILE* file = fopen("results/results_Schedule.txt", "w");
        fprintf(file, "%-10s %15s %15s\n", "Type schedule",  "Chunks", "Time");
        results.resize(schedules.size() * chunks.size());
        int i = 0;

        for(std::string schedule : schedules)
        {   
            printf("Type: %10s\n", schedule.c_str());
            for (int chunk : chunks)
            {
                std::fill(x.begin(), x.end(), 0.0);
                results[i] = run_parallel(A, x, b, 8, schedule, chunk);
                printf("Chunks: %d | Time: %.6f sec\n", chunk, results[i]);
                fprintf(file, "%10s %15.d %20.6f\n", schedule.c_str(), chunk, results[i]);
                ++i;
            }
            fprintf(file, "\n");
            printf("\n");
        }
        fclose(file);
    }

    
    if(flag == 1){
        FILE* file = fopen("results/results1_2.txt", "w");
        fprintf(file, "%-10s %15s %15s\n", "Threads(I)", "Time(sec)(T_i)", "Gain(S)");
        
        for (size_t i = 0; i < list_threads.size(); i++) {
            double gain =  results[0] / results[i];
            fprintf(file, "%10d %15.6f %15.6f\n", list_threads[i], results[i], gain);
        }
        fclose(file);
        printf("Results saved to results.txt\n");
    } 

    return 0;
}