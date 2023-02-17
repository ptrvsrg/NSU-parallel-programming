#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 5000
#define EPSILON 1E-7
#define TAU 1E-5
#define MAX_ITERATION_COUNT 100000

void SetMatrixPart(int *lineCounts, int *lineOffsets, int size, int processCount);
void GenerateAChunks(double *AChunk, int lineCount, int lineSize, int processRank);
void GenerateX(double *x, int size);
void GenerateB(double *b, int size);
double CalcNormSquare(const double *vector, int size);
void CalcAxb(const double* AChunk, const double* x, const double* b, 
             double* AxbChunk, int chunkSize, int chunkOffset);
void CalcNextX(const double* AxbChunk, const double* x, double* xChunk, 
               double tau, int chunkSize, int chunkOffset);
// void PrintMatrix(const double *a, int lines, int columns);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int processRank;
    int processCount;
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    
    int *lineCounts = malloc(sizeof(int) * processCount);
    int *lineOffsets = malloc(sizeof(int) * processCount);
    SetMatrixPart(lineCounts, lineOffsets, N, processCount);

    double *AChunk = malloc(sizeof(double) * lineCounts[processRank] * N);
    double *x = malloc(sizeof(double) * N);
    double *b = malloc(sizeof(double) * N);
    GenerateAChunks(AChunk, lineCounts[processRank], N, lineOffsets[processRank]);
    GenerateX(x, N);
    GenerateB(b, N);

    double bNorm;
    if (processRank == 0)
        bNorm = sqrt(CalcNormSquare(b, N));

    double *AxbChunk = malloc(sizeof(double) * lineCounts[processRank]);
    double *xChunk = malloc(sizeof(double) * lineCounts[processRank]);
    double accuracy = EPSILON + 1;
    int iterationCount = 0;
    double startTime = MPI_Wtime();
    double finishTime;

    while (accuracy > EPSILON && iterationCount < MAX_ITERATION_COUNT)
    {
        CalcAxb(AChunk, x, b, AxbChunk, 
                lineCounts[processRank], lineOffsets[processRank]);
  
        CalcNextX(AxbChunk, x, xChunk, TAU, lineCounts[processRank], lineOffsets[processRank]);
        MPI_Allgatherv(xChunk, lineCounts[processRank], MPI_DOUBLE,
                       x, lineCounts, lineOffsets, MPI_DOUBLE, MPI_COMM_WORLD);

        double AxbChunkNormSquare = CalcNormSquare(AxbChunk, lineCounts[processRank]);
        MPI_Reduce(&AxbChunkNormSquare, &accuracy, 1, 
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (processRank == 0)
            accuracy = sqrt(accuracy) / bNorm;
        MPI_Bcast(&accuracy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        ++iterationCount;
    }

    finishTime = MPI_Wtime();

    if (processRank == 0)
    {
        if (iterationCount == MAX_ITERATION_COUNT)
            printf("Too many iterations\n");
        else
        {
            printf("Norm: %lf\n", sqrt(CalcNormSquare(x, N)));
            printf("Time: %lf sec\n", finishTime - startTime);
        }
    }

    free(lineCounts);
    free(lineOffsets);
    free(x);
    free(b);
    free(AChunk);
    free(AxbChunk);
    free(xChunk);

    MPI_Finalize();

    return 0;
}

void GenerateAChunks(double *AChunk, int lineCount, int lineSize, int lineIndex)
{
    for (int i = 0; i < lineCount; i++)
    {
        for (int j = 0; j < lineSize; ++j)
            AChunk[i * lineSize + j] = 1;

        AChunk[i * lineSize + lineIndex + i] = 2;
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

void SetMatrixPart(int* lineCounts, int* lineOffsets, int size, int processCount) 
{
    int offset = 0;
    for (int i = 0; i < processCount; ++i)
    {
        lineCounts[i] = size / processCount;
        
        if (i < size % processCount)
            ++lineCounts[i];

        lineOffsets[i] = offset;
        offset += lineCounts[i];
    }
}

double CalcNormSquare(const double *vector, int size)
{
    double secondNormSquare = 0.0;
    for (int i = 0; i < size; ++i)
        secondNormSquare += vector[i] * vector[i];

    return secondNormSquare;
}

void CalcAxb(const double* AChunk, const double* x, const double* b, double* AxbChunk, int chunkSize, int chunkOffset) 
{
    for (int i = 0; i < chunkSize; ++i)
    {
        AxbChunk[i] = -b[chunkOffset + i];
        for (int j = 0; j < N; ++j)
            AxbChunk[i] += AChunk[i * N + j] * x[j];
    }
}

void CalcNextX(const double* AxbChunk, const double* x, double* xChunk, double tau, int chunkSize, int chunkOffset) 
{
    for (int i = 0; i < chunkSize; ++i)
        xChunk[i] = x[chunkOffset + i] - tau * AxbChunk[i];
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
