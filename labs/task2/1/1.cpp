#include <iostream>
#include <omp.h>
#include <cstdio>
#include <vector>
#include <inttypes.h> 
#include <cstdio>
#include <string>

#define M 40000
#define N 40000

using TYPE = std::vector<double>;

void init(TYPE& a, TYPE& b) 
{
    a.resize(M * N);
    b.resize(N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++)
            a[i * N + j] = static_cast<double>(i + j);
    }
    for (int j = 0; j < N; j++)
        b[j] = static_cast<double>(j);
}

void matrix_vector_product_omp(const TYPE& a, const TYPE& b, TYPE& c, int num_threads)
{
    #pragma omp parallel num_threads(num_threads)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();

        int items_per_thread = M / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (M - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; i++) {
            c[i] = 0.0;
            for (int j = 0; j < N; j++)
                c[i] += a[i * N + j] * b[j];
        }   
    }
}

double run_parallel(const TYPE& a, const TYPE& b, TYPE& c, int num_threads) 
{
    double t = omp_get_wtime();
    matrix_vector_product_omp(a, b, c, num_threads);
    t = omp_get_wtime() - t;
    return t;
}

int main() 
{
    TYPE a, b, c;
    init(a, b);
    c.resize(M);

    std::vector<int> list_threads = {1, 2, 4, 6, 8, 16, 20, 40};
    TYPE results;
    results.resize(list_threads.size());
    

    printf("Matrix-vector product (c[M] = a[M, N] * b[N]; M = %d, N = %d)\n", M, N);
    printf("Memory used: %" PRIu64 " MiB\n", (M * N + M + N) * sizeof(double) / (1024 * 1024));
    
    for (size_t i = 0; i < list_threads.size(); i++){
        results[i] = run_parallel(a, b, c, list_threads[i]);
    }

    FILE* file = fopen("results/results2.txt", "w");
    if (file) {
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