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
void CalcAxb(const double *AChunk, const double *xChunk, const double *bChunk, double *recvXChunk, 
             double *AxbChunk, int *lineCounts, int *lineOffsets, int processRank, int processCount);
void CalcNextX(const double *Axb, double *xChunk, double tau, int chunkSize);
void CopyMatrix(double *dest, const double *src, int lines, int columns);
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
    double *recvXChunk = malloc(sizeof(double) * lineCounts[0]);
    double accuracy = EPSILON + 1;
    int iterationCount = 0;
    double startTime = MPI_Wtime();
    double finishTime;

    while (accuracy > EPSILON && iterationCount < MAX_ITERATION_COUNT)
    {
        CopyMatrix(recvXChunk, xChunk, lineCounts[processRank], 1);
        CalcAxb(AChunk, xChunk, bChunk, recvXChunk, AxbChunk, 
                lineCounts, lineOffsets, processRank, processCount);

        CalcNextX(AxbChunk, xChunk, TAU, lineCounts[processRank]);

        double AxbChunkNormSquare = CalcNormSquare(AxbChunk, lineCounts[processRank]);
        MPI_Reduce(&AxbChunkNormSquare, &accuracy, 1, 
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (processRank == 0)
            accuracy = sqrt(accuracy) / bNorm;
        MPI_Bcast(&accuracy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        ++iterationCount;
    }

    finishTime = MPI_Wtime();

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

void CalcAxb(const double *AChunk, const double *xChunk, const double *bChunk, double *recvXChunk, 
             double *AxbChunk, int *lineCounts, int *lineOffsets, int processRank, int processCount)
{
    int srcRank = (processRank + processCount - 1) % processCount;
    int destRank = (processRank + 1) % processCount;

    for (int i = 0; i < processCount; ++i)
    {
        int currentRank = (processRank + i) % processCount;
        for (int j = 0; j < lineCounts[processRank]; ++j)
        {
            if (i == 0)
                AxbChunk[j] = -bChunk[j];
            for (int k = 0; k < lineCounts[currentRank]; ++k)
                AxbChunk[j] += AChunk[j * N + lineOffsets[currentRank] + k] * recvXChunk[k];
        }

        if (i != processCount - 1)
            MPI_Sendrecv_replace(recvXChunk, lineCounts[0], MPI_DOUBLE, destRank, processRank, 
                                 srcRank, srcRank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void CalcNextX(const double *AxbChunk, double *xChunk, double tau, int chunkSize)
{
    for (int i = 0; i < chunkSize; ++i)
        xChunk[i] -= tau * AxbChunk[i];
}

void CopyMatrix(double *dest, const double *src, int lines, int columns)
{
    for (int i = 0; i < lines; i++)
        for (int j = 0; j < columns; j++)
            dest[i * columns + j] = src[i * columns + j];
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
