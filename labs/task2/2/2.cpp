#include <iostream>
#include <omp.h>
#include <cstdio>
#include <vector>
#include <inttypes.h> 
#include <cstdio>
#include <math.h>


#define PI 3.14159265358979323846
#define A -4.0
#define B 4.0
#define N_STEPS 80000000
const double h =  (B - A) / N_STEPS;


double func(double x)
{
    return exp(-x * x);
}


double integrate()
{
    double sum = 0.0;

    for (int i = 0; i < N_STEPS; i++)
        sum += func(A + h * (i + 0.5));

    sum *= h;

    return sum;
}

double integrate_omp(int num_threads)
{
    double sum = 0.0;

    #pragma omp parallel num_threads(num_threads)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = N_STEPS / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (N_STEPS - 1) : (lb + items_per_thread - 1);
        double sumloc = 0.0;

        for (int i = lb; i <= ub; i++)
            sumloc += func(A + h * (i + 0.5));
        
        #pragma omp atomic
        sum += sumloc;
    }

    sum *= h;
    return sum;
}

double run_parallel(int num_threads)
{
    double t = omp_get_wtime();
    double res = integrate_omp(num_threads);
    t = omp_get_wtime() - t;
    return t;
}

int main()
{
    std::vector<int> list_threads = {1, 2, 4, 6, 8, 16, 20, 40};
    std::vector<double> results;
    results.resize(list_threads.size());
    

    printf("Integration f(x) on [%.12f, %.12f], N_STEPS = %d\n", A, B, N_STEPS);
    
    for (size_t i = 0; i < list_threads.size(); i++){
        results[i] = run_parallel(list_threads[i]);
    }

    FILE* file = fopen("results/results2.txt", "w");
    if (file) {
        fprintf(file, "%-10s %15s %15s\n", "Threads(I)", "Time(sec)(T_i)", "Gain(S)");
        
        for (size_t i = 0; i < list_threads.size(); i++) {
            double gain =  results[0] / results[i];
            fprintf(file, "%10d %15.6f %15.6f\n", list_threads[i], results[i], gain);
        }
        fclose(file);
        printf("Results saved to results2.txt\n");
    }
    return 0;
}