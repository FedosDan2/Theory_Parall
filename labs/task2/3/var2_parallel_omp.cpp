#include <iostream>
#include <omp.h>
#include <cstdio>
#include <vector>
#include <inttypes.h> 
#include <cmath>
#include <string>

#define EPS 1e-5
#define MAX_ITER 10000
#define N 1000

using TYPE = std::vector<double>;

void init_paral(TYPE& A, int lb, int ub) 
{
    for (int i = lb; i < ub; i++) {
        for (int j = 0; j < N; j++){
            if(i == j) 
                A[i * N + j] = 2.0;
            else 
                A[i * N + j] = 1.0;
        }
    } 
}


// Вариант 2: Cоздается одна параллельная секция #pragma omp parallel, охватывающая весь итерационный алгоритм
void lin_equation_omp2(TYPE& A, TYPE& x, const TYPE& b, int num_threads)
{
    double tau = 0.01;
    TYPE Ax(N);
    #pragma omp parallel num_threads(num_threads)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();

        int items_per_thread = N / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (N - 1) : (lb + items_per_thread - 1);

        init_paral(A, lb, ub);
        
        for (int iter = 0; iter < MAX_ITER; iter++) {
            std::fill(Ax.begin(), Ax.end(), 0.0);
            
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
    
            bool converged = false;
            #pragma omp single
            {
                if (std::sqrt(global_diff_norm) < EPS) {
                    converged = true;
                }
            }
            #pragma omp barrier
            if (converged) break;
        }  
    }
}

double run_parallel(TYPE& A, TYPE& x, const TYPE& b, int num_threads) 
{   
    double t = omp_get_wtime();
    lin_equation_omp2(A, x, b, num_threads);
    t = omp_get_wtime() - t;
    return t;
}

int main() 
{
    TYPE A(N * N), b(N);
    TYPE x(N, 0.0);

    for (int j = 0; j < N; j++)
        b[j] = static_cast<double>(N+1);

    std::vector<int> list_threads = {1, 2, 4, 6, 8, 16, 20, 40};
    std::vector<double> results(list_threads.size());

    printf("Linear equation - Variant 2 (N=%d)\n", N);
    printf("Memory: %" PRIu64 " MiB\n", static_cast<uint64_t>(N * N + 2 * N) * sizeof(double) / (1024 * 1024));

    const int runs = 5;
    
    for (size_t i = 0; i < list_threads.size(); i++) {
        double total_time = 0.0;
        
        for (int r = 0; r < runs; r++) {
            std::fill(x.begin(), x.end(), 0.0);
            total_time += run_parallel(A, x, b, list_threads[i]);
        }
        results[i] = total_time / runs;
        printf("Threads %2d: %.4f sec\n", list_threads[i], results[i]);
    }
    
    FILE* file = fopen("results/var2/results2.txt", "w");
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