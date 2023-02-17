#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 5000
#define EPSILON 1E-7
#define TAU 1E-5
#define MAX_ITERATION_COUNT 1000000

void SetMatrixPart(int *lineCounts, int *lineOffsets, int size, int processCount);
void GenerateAChunk(double *AChunk, int lineCount, int lineSize, int processRank);
void GenerateXChunk(double *xChunk, int size);
void GenerateBChunk(double *bChunk, int size);
double CalcNormSquare(double *vector, int size);
void CalcAxb(const double *AChunk, const double *x, const double *bChunk, 
             double *AxbChunk, int chunkSize);
void CalcNextX(const double *Axb, double *xChunk, double tau, int chunkSize);
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

    double *xChunk = malloc(sizeof(double) * lineCounts[processRank]);
    double *bChunk = malloc(sizeof(double) * lineCounts[processRank]);
    double *AChunk = malloc(sizeof(double) * lineCounts[processRank] * N);

    GenerateXChunk(xChunk, lineCounts[processRank]);
    GenerateBChunk(bChunk, lineCounts[processRank]);
    GenerateAChunk(AChunk, lineCounts[processRank], N, lineOffsets[processRank]);

    double bChunkNorm = CalcNormSquare(bChunk, lineCounts[processRank]);
    double bNorm;
    MPI_Reduce(&bChunkNorm, &bNorm, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);
    if (processRank == 0)
        bNorm = sqrt(bNorm);

    double *AxbChunk = malloc(sizeof(double) * lineCounts[processRank]);
    double *x = malloc(sizeof(double) * N);
    double accuracy = EPSILON + 1;
    int iterationCount = 0;
    double startTime = MPI_Wtime();
    double finishTime;

    while (accuracy > EPSILON && iterationCount < MAX_ITERATION_COUNT)
    {
        MPI_Allgatherv(xChunk, lineCounts[processRank], MPI_DOUBLE,
                       x, lineCounts, lineOffsets, MPI_DOUBLE, MPI_COMM_WORLD);
        CalcAxb(AChunk, x, bChunk, AxbChunk, lineCounts[processRank]);

        CalcNextX(AxbChunk, xChunk, TAU, lineCounts[processRank]);

        double AxbChunkNormSquare = CalcNormSquare(AxbChunk, lineCounts[processRank]);
        MPI_Reduce(&AxbChunkNormSquare, &accuracy, 1, MPI_DOUBLE,
                   MPI_SUM, 0, MPI_COMM_WORLD);
        if (processRank == 0)
            accuracy = sqrt(accuracy) / bNorm;
        MPI_Bcast(&accuracy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        ++iterationCount;
    }

    finishTime = MPI_Wtime();

    PrintMatrix(xChunk, 1, lineCounts[processRank]);
    MPI_Barrier(MPI_COMM_WORLD);

    double xChunkNorm = CalcNormSquare(xChunk, lineCounts[processRank]);
    double xNorm;
    MPI_Reduce(&xChunkNorm, &xNorm, 1, MPI_DOUBLE,
                MPI_SUM, 0, MPI_COMM_WORLD);
    if (processRank == 0)
    {
        if (iterationCount == MAX_ITERATION_COUNT)
            fprintf(stderr, "Too many iterations\n");
        else
        {
            printf("Norm: %lf\n", sqrt(xNorm));
            printf("Time: %lf sec\n", finishTime - startTime);
        }
    }

    free(lineCounts);
    free(lineOffsets);
    free(xChunk);
    free(bChunk);
    free(AChunk);
    free(AxbChunk);
    free(x);

    MPI_Finalize();

    return 0;
}

void GenerateAChunk(double *AChunk, int lineCount, int lineSize, int lineIndex)
{
    for (int i = 0; i < lineCount; i++)
    {
        for (int j = 0; j < lineSize; ++j)
            AChunk[i * lineSize + j] = 1;

        AChunk[i * lineSize + lineIndex + i] = 2;
    }
}

void GenerateXChunk(double *xChunk, int size)
{
    for (int i = 0; i < size; i++)
        xChunk[i] = 0;
}

void GenerateBChunk(double *bChunk, int size)
{
    for (int i = 0; i < size; i++)
        bChunk[i] = N + 1;
}

void SetMatrixPart(int *lineCounts, int *lineOffsets, int size, int processCount)
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

double CalcNormSquare(double *vector, int size)
{
    double secondNormSquare = 0.0;
    for (int i = 0; i < size; ++i)
        secondNormSquare += vector[i] * vector[i];

    return secondNormSquare;
}

void CalcAxb(const double *AChunk, const double *x, const double *bChunk, 
             double *AxbChunk, int chunkSize)
{
    for (int i = 0; i < chunkSize; ++i)
    {
        AxbChunk[i] = -bChunk[i];
        for (int j = 0; j < N; ++j)
            AxbChunk[i] += AChunk[i * N + j] * x[j];
    }
}

void CalcNextX(const double *AxbChunk, double *xChunk, double tau, int chunkSize)
{
    for (int i = 0; i < chunkSize; ++i)
        xChunk[i] -= tau * AxbChunk[i];
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

// void CalcAxb(const double *AChunk, const double *x, const double *bChunk, 
//              double *AxbChunk, int chunkSize)
// {
//     for (int i = 0; i < lineCounts[processRank]; ++i)
//     {
//         AxbChunk[i] = -bChunk[i];

//         for (int j = 0; j < lineCounts[processRank]; ++j)
//             AxbChunk[i] += AChunk[i * N + lineOffsets[processRank] + j] * xChunk[j];
//     }

//     int srcRank = (processRank + processCount - 1) % processCount;
//     int destRank = (processRank + 1) % processCount;

//     for (int i = 0; i < processCount - 1; ++i)
//     {
//         MPI_Sendrecv(xChunk, lineCounts[processRank], MPI_DOUBLE, destRank, 0,
//                      recvXChunk, lineCounts[srcRank], MPI_DOUBLE, srcRank, 0,
//                      MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//         for (int j = 0; j < lineCounts[processRank]; ++j)
//         {
//             AxbChunk[j] = -bChunk[j];

//             for (int k = 0; k < lineCounts[srcRank]; ++k)
//                 AxbChunk[j] += AChunk[j * N + lineOffsets[srcRank] + k] * recvXChunk[k];
//         }

//         srcRank = (srcRank + processCount - 1) % processCount;
//         destRank = (destRank + 1) % processCount;
//     }
// }