#include <iostream>
#include <omp.h>
#include <cstdio>
#include <vector>
#include <inttypes.h> 
#include <string>

#define M 40000
#define N 40000

using TYPE = std::vector<double>;
// метрика эффективности и графики разбить для понятного восприятия
void init(TYPE& a, int lb, int ub) 
{
    for (int i = lb; i <= ub; i++) {
        for (int j = 0; j < N; j++)
            a[i * N + j] = static_cast<double>(i + j);
    }
}

// смысл паральлельной инициализации для основного кода
void matrix_vector_product_omp(TYPE& a, TYPE& b, TYPE& c, int num_threads)
{
    #pragma omp parallel num_threads(num_threads)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();

        int items_per_thread = M / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (M - 1) : (lb + items_per_thread - 1);
        
        // паралельная инициализация
        init(a, lb, ub);

        for (int i = lb; i <= ub; i++) {
            c[i] = 0.0;
            for (int j = 0; j < N; j++)
                c[i] += a[i * N + j] * b[j];
        }   
    }
}

double run_parallel(TYPE& a, TYPE& b, TYPE& c, int num_threads) 
{   
    for (int j = 0; j < N; j++)
        b[j] = static_cast<double>(j);

    double t = omp_get_wtime();
    matrix_vector_product_omp(a, b, c, num_threads);
    t = omp_get_wtime() - t;
    return t;
}

int main() 
{
    TYPE a, b, c;
    a.resize(M * N);
    b.resize(N);
    c.resize(M);

    std::vector<int> list_threads = {1, 2, 4, 6, 8, 16, 20, 40};
    TYPE results;
    results.resize(list_threads.size());
    

    printf("Matrix-vector product (c[M] = a[M, N] * b[N]; M = %d, N = %d)\n", M, N);
    printf("Memory used: %" PRIu64 " MiB\n", (M * N + M + N) * sizeof(double) / (1024 * 1024));
    
    int create_average_result = 20;
    // запускаем на одно несколько раз чтобы получить среднее
    for (size_t i = 0; i < list_threads.size(); i++){
        double average_results_thread = 0.0;

        for (size_t j = 0; j < create_average_result; j++){
            average_results_thread += run_parallel(a, b, c, list_threads[i]);
        }
        results[i] = average_results_thread / create_average_result;
    }

    FILE* file = fopen("results/results2.txt", "w");
    if (file) {
        fprintf(file, "%s %d %s\n", "Average Results after", create_average_result, "operations:");
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