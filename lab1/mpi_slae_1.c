#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 10

void SetMatrixPart(int *lineCounts, int *lineOffsets, int size, int processCount);
void GenerateAChunks(double *AChunk, int lineCount, int lineSize, int processRank);
void GenerateX(double *x, int size);
void GenerateB(double *b, int size);
double CalcNorm(double *vector, int size);
void CalcAxb(const double* AChunk, const double* x, const double* b, double* AxbBuff, int partSize, int partOffset);
void CalcNextX(const double* Axb, const double* x, double* xBuff, double tau, int partSize, int partOffset);
void PrintMatrix(double *a, int lines, int columns);

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int processRank;
    int processCount;
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processRank);
    
    // Initialize counts of lines in matrix parts and begin indexes of lines in matrix parts
    int *lineCounts = malloc(sizeof(int) * processCount);
    int *lineOffsets = malloc(sizeof(int) * processCount);
    SetMatrixPart(lineCounts, lineOffsets, N, processCount);

    // Allocate and initialize parts of matrix A
    double *AChunk = malloc(sizeof(double) * lineCounts[processRank] * N);  
    GenerateAChunks(AChunk, lineCounts[processRank], N, lineOffsets[processRank]);

    // Allocate and initialize vectors b, x 
    double *x = malloc(sizeof(double) * N);
    double *b = malloc(sizeof(double) * N);
    if (processRank == 0)
    {
        GenerateB(b, N);
        GenerateX(x, N);
    }
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Calculate second norm of vector b
    double bNorm = CalcNorm(b, N);

    // Create vector Axb for process 0
    double *Axb = NULL;
    if (processRank == 0)
        Axb = malloc(sizeof(double) * N);

    // Create buffers
    double *AxbBuff = malloc(sizeof(double) * N);
    double *xBuff = malloc(sizeof(double) * N);

    // Create some parameters
    double e = 0.000001;
    double tau = 0.001;
    double accuracy = e + 1;

    // Start counting time
    double startTime = MPI_Wtime();

    while (accuracy > e)
    {
        // Calculate parts of vector A*x-b
        CalcAxb(AChunk, x, b, AxbBuff, lineCounts[processRank], lineOffsets[processRank]);
        MPI_Gatherv(AxbBuff + lineOffsets[processRank], lineCounts[processRank], MPI_DOUBLE,
                    Axb, lineCounts, lineOffsets, MPI_DOUBLE, 
                    0, MPI_COMM_WORLD);

        // Calculate next vector x          
        CalcNextX(AxbBuff, x, xBuff, tau, lineCounts[processRank], lineOffsets[processRank]);
        MPI_Allgatherv(xBuff + lineOffsets[processRank], lineCounts[processRank], MPI_DOUBLE,
                       x, lineCounts, lineOffsets, MPI_DOUBLE, MPI_COMM_WORLD);

        // Update accuracy and broadcasts accuracy to all processes
        if (processRank == 0)
            accuracy = CalcNorm(Axb, N) / bNorm;
        MPI_Bcast(&accuracy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    // Finish counting time
    double finishTime = MPI_Wtime();

    // Print result
    if (processRank == 0)
    {
        printf("Norm: %lf\n", pow(CalcNorm(x, N), 2));
        printf("Time: %lf sec\n", finishTime - startTime);
    }

    // Free up memory
    free(lineCounts);
    free(lineOffsets);
    free(AChunk);
    free(x);
    free(b);
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
        x[i] = ((double) rand() / RAND_MAX) * 100;
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
        AxbBuff[partOffset + i] = -b[partOffset + i];
        for (int j = 0; j < N; ++j)
            AxbBuff[partOffset + i] += AChunk[i * N + j] * x[j];
    }
}

void CalcNextX(const double* AxbBuff, const double* x, double* xBuff, double tau, int partSize, int partOffset) 
{
    for (int i = 0; i < partSize; ++i)
        xBuff[partOffset + i] = x[partOffset + i] - tau * AxbBuff[partOffset + i];
}

void PrintMatrix(double *a, int lines, int columns)
{
    for (int i = 0; i < lines; i++)
    {
        for (int j = 0; j < columns; j++)
            printf("%lf ", a[i * columns + j]);

        printf("\n");
    }
}
