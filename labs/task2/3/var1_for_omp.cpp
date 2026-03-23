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

    // Заполнение матрицы
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
void lin_equation_omp1(const TYPE& A, TYPE& x, const TYPE& b, int num_threads)
{   
    double tau = 0.01;
    TYPE Ax(N);
    for (int iter = 0; iter < MAX_ITER; iter++) {
        std::fill(Ax.begin(), Ax.end(), 0.0);
        
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
        
        if (std::sqrt(global_diff_norm) < EPS) {
            break;
        }
    }
}

double run_parallel(const TYPE& A, TYPE& x, TYPE& b, int num_threads) 
{   
    double t = omp_get_wtime();
    lin_equation_omp1(A, x, b, num_threads);
    t = omp_get_wtime() - t;
    return t;
}

int main() 
{
    TYPE A, b;
    TYPE x(N, 0.0);   
    
    init(A, b);
    
    std::vector<int> list_threads = {1, 2, 4, 6, 8, 16, 20, 40};
    std::vector<double> results(list_threads.size());

    printf("Linear equation (N=%d)\n", N);
    printf("Memory: %" PRIu64 " MiB\n", static_cast<uint64_t>(N * N + 2 * N) * sizeof(double) / (1024 * 1024));

    const int runs = 5;  // Усреднение
    for (size_t i = 0; i < list_threads.size(); i++) {
        double total_time = 0.0;
        
        for (int r = 0; r < runs; r++) {
            std::fill(x.begin(), x.end(), 0.0);  // Сброс перед каждым запуском
            total_time += run_parallel(A, x, b, list_threads[i]);
        }
        results[i] = total_time / runs;
        printf("Threads %2d: %.4f sec\n", list_threads[i], results[i]);
    }
    
    FILE* file = fopen("results/var1/results2.txt", "w");
    if (file) {
        fprintf(file, "%s %d %s\n", "Average Results after", runs, "operations:");
        fprintf(file, "%10s %15s %15s %15s\n", "Threads(I)", "Time(sec)(T_i)", "Gain(S)", "Gain Modified(S)");
        
        for (size_t i = 0; i < list_threads.size(); i++) {
            double gain =  results[0] / results[i];
            double gain_modified = gain / list_threads[i];
            fprintf(file, "%10d %15.6f %15.6f %15.6f\n", list_threads[i], results[i], gain, gain_modified);
        }
        fclose(file);
        printf("Results saved to results.txt\n");
    }
    return 0;
}