#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000

void SetMatrixPart(int *lineCounts, int *lineOffsets, int size, int processCount);
void GenerateAChunks(double *AChunk, int lineCount, int lineSize, int processRank);
void GenerateX(double *x, int size);
void GenerateB(double *b, int size);
double CalcNorm(double *vector, int size);
void CalcAxb(const double* AChunk, const double* x, const double* b, double* AxbBuff, int partSize, int partOffset);
void CalcNextX(const double* Axb, const double* x, double* xBuff, double tau, int partSize, int partOffset);
void PrintMatrix(const double *a, int lines, int columns);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    double e = 0.000001;
    double tau = 0.001;
    double accuracy = e + 1;
    double bNorm;
    double startTime;
    double finishTime;

    int processRank;
    int processCount;
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    
    int *lineCounts = malloc(sizeof(int) * processCount);
    int *lineOffsets = malloc(sizeof(int) * processCount);
    SetMatrixPart(lineCounts, lineOffsets, N, processCount);

    double *x = malloc(sizeof(double) * N);
    double *b = malloc(sizeof(double) * N);
    double *AChunk = malloc(sizeof(double) * lineCounts[processRank] * N);  
    double *Axb = NULL;
    double *AxbBuff = malloc(sizeof(double) * lineCounts[processRank]);
    double *xBuff = malloc(sizeof(double) * lineCounts[processRank]);

    if (processRank == 0)
    {
        GenerateX(x, N);
        GenerateB(b, N);
        Axb = malloc(sizeof(double) * N);
    }
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    GenerateAChunks(AChunk, lineCounts[processRank], N, lineOffsets[processRank]);

    bNorm = CalcNorm(b, N);
    startTime = MPI_Wtime();

    while (accuracy > e)
    {
        CalcAxb(AChunk, x, b, AxbBuff, lineCounts[processRank], lineOffsets[processRank]);
        MPI_Gatherv(AxbBuff, lineCounts[processRank], MPI_DOUBLE,
                    Axb, lineCounts, lineOffsets, MPI_DOUBLE, 
                    0, MPI_COMM_WORLD);
  
        CalcNextX(AxbBuff, x, xBuff, tau, lineCounts[processRank], lineOffsets[processRank]);
        MPI_Allgatherv(xBuff, lineCounts[processRank], MPI_DOUBLE,
                       x, lineCounts, lineOffsets, MPI_DOUBLE, MPI_COMM_WORLD);

        if (processRank == 0)
            accuracy = CalcNorm(Axb, N) / bNorm;
        MPI_Bcast(&accuracy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    finishTime = MPI_Wtime();

    if (processRank == 0)
    {
        printf("Norm: %lf\n", pow(CalcNorm(x, N), 2));
        printf("Time: %lf sec\n", finishTime - startTime);
    }

    free(lineCounts);
    free(lineOffsets);
    free(x);
    free(b);
    free(AChunk);
    free(AxbBuff);
    free(xBuff);

    if (processRank == 0)
        free(Axb);

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
    srand(time(NULL));
    for (int i = 0; i < size; i++)
        x[i] = ((double) rand() / RAND_MAX) * rand();
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

double CalcNorm(double *vector, int size)
{
    double secondNormSquare = 0.0;
    for (int i = 0; i < size; ++i)
        secondNormSquare += vector[i] * vector[i];

    return sqrt(secondNormSquare);
}

void CalcAxb(const double* AChunk, const double* x, const double* b, double* AxbBuff, int partSize, int partOffset) 
{
    for (int i = 0; i < partSize; ++i)
    {
        AxbBuff[i] = -b[partOffset + i];
        for (int j = 0; j < N; ++j)
            AxbBuff[i] += AChunk[i * N + j] * x[j];
    }
}

void CalcNextX(const double* AxbBuff, const double* x, double* xBuff, double tau, int partSize, int partOffset) 
{
    for (int i = 0; i < partSize; ++i)
        xBuff[i] = x[partOffset + i] - tau * AxbBuff[i];
}

void PrintMatrix(const double *a, int lines, int columns)
{
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < columns; j++)
            printf("%lf ", a[i * columns + j]);

        printf("\n");
    }
}
