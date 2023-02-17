#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 5000
#define EPSILON 1E-7
#define TAU 1E-5
#define MAX_ITERATION_COUNT 1000000

void GenerateA(double *A, int size);
void GenerateX(double *x, int size);
void GenerateB(double *b, int size);
double CalcNormSquare(const double *vector, int size);
void CalcAxb(const double* A, const double* x, const double* b, double* Axb, int size);
void CalcNextX(const double* Axb, double* x, double tau, int size);
// void PrintMatrix(const double *a, int lines, int columns);

int main(int argc, char **argv)
{
    double *A = malloc(sizeof(double) * N * N);
    double *x = malloc(sizeof(double) * N);
    double *b = malloc(sizeof(double) * N);
    GenerateA(A, N);
    GenerateX(x, N);
    GenerateB(b, N);

    double bNorm = sqrt(CalcNormSquare(b, N));

    double *Axb = malloc(sizeof(double) * N);
    double accuracy = EPSILON + 1;
    int iterationCount = 0;
    struct timespec startTime;
    struct timespec finishTime;
    clock_gettime(CLOCK_MONOTONIC_RAW, &startTime);

    while (accuracy > EPSILON && iterationCount < MAX_ITERATION_COUNT)
    {
        CalcAxb(A, x, b, Axb, N);
        CalcNextX(Axb, x, TAU, N);
        accuracy = sqrt(CalcNormSquare(Axb, N)) / bNorm;
        ++iterationCount;
    }

    clock_gettime(CLOCK_MONOTONIC_RAW, &finishTime);

    if (iterationCount == MAX_ITERATION_COUNT)
        printf("Too many iterations\n");
    else
    {
        printf("Norm: %lf\n", sqrt(CalcNormSquare(x, N)));
        printf("Time: %lf sec\n", finishTime.tv_sec - startTime.tv_sec + 1E-09 * (finishTime.tv_nsec - startTime.tv_nsec));
    }

    free(A);
    free(x);
    free(b);
    free(Axb);

    return 0;
}

void GenerateA(double *A, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; ++j)
            A[i * size + j] = 1;

        A[i * size + i] = 2;
    }
}

void GenerateX(double *x, int size)
{
    for (int i = 0; i < size; i++)
        x[i] = 0;
}

void GenerateB(double *b, int size)
{
    for (int i = 0; i < size; i++)
        b[i] = N + 1;
}

double CalcNormSquare(const double *vector, int size)
{
    double secondNormSquare = 0.0;
    for (int i = 0; i < size; ++i)
        secondNormSquare += vector[i] * vector[i];

    return secondNormSquare;
}

void CalcAxb(const double* A, const double* x, const double* b, double* Axb, int size) 
{
    for (int i = 0; i < size; ++i)
    {
        Axb[i] = -b[i];
        for (int j = 0; j < N; ++j)
            Axb[i] += A[i * N + j] * x[j];
    }
}

void CalcNextX(const double* Axb, double* x, double tau, int size) 
{
    for (int i = 0; i < size; ++i)
        x[i] -= tau * Axb[i];
}

// void PrintMatrix(const double *a, int lines, int columns)
// {
//     for (int i = 0; i < lines; i++)
//     {
//         for (int j = 0; j < columns; j++)
//             printf("%lf ", a[i * columns + j]);

//         printf("\n");
//     }
// }
