#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <omp.h>


// Tamanho do bloco de threads
#define BLOCK_SIZE 32

// Definindo o tipo MATRIZ
typedef struct mat{
    int m;
    int n;
    int des;
    double *dados;
}MATRIX;

//######################## Funcoes ###############################
// Funcao para preencher a matriz com 0 ('z'), 1('o') ou rand (qualquer outro)
void preencher_mat (MATRIX mat, char type){
    int i;
    for (i=0; i < mat.m * mat.n; i++){
        if (type == 'z')
            mat.dados[i] = 0.0;
        else if (type == 'o')
            mat.dados[i] = 1.0;
        else
            mat.dados[i] = (rand() % 10) / 10.0;
    }
}

// Imprimir a matriz toda ou um pedaço tam x tam da matriz
void print_mat (MATRIX mat, int tam){
    int lim1, lim2, i, j;
    if (tam == -1){
        lim1 = mat.m;
        lim2 = mat.n;
    } else{
        lim1 = tam;
        lim2 = tam;
    }
    for(i = 0; i < lim1; i++){
        for(j = 0; j < lim2; j++){
            printf("%f ", mat.dados[i * mat.n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Funcao para multiplicar matriz na CPU sem ladrilho
void mult_mat_cpu (MATRIX A, MATRIX B, MATRIX C){
    int i, j, k;

    #pragma omp paralell private (i,j,k) shared (A,B,C)
    {
        #pragma omp paralell schedule (static)
        for (i=0; i<A.m; i++){
            for (j=0; j<B.n; j++){
                double soma = 0;
                for (k=0; k<A.n; k++){
                    soma += A.dados[i * A.n + k] * B.dados[k * B.n + j];
                }
                C.dados[i * A.n + j] = soma;
            }
        }
    }
}

// Funcao para multiplicar matriz na CPU com ladrilho
void mult_mat_cpu_ladrilhada(MATRIX A,MATRIX B, MATRIX C){
	int i,j,k,x,y,z;

    #pragma omp paralell default(none) private (i,j,k,x,y,z) shared (A,B,C)
    {
        for (i = 0; i < C.m; i += BLOCK_SIZE) {
             for (j = 0; j < C.n; j += BLOCK_SIZE) {
                 #pragma omp for schedule (static)
                 for (k = 0; k < C.n; k += BLOCK_SIZE) {
                     for (x = i; x < (i + BLOCK_SIZE); x++) {
                         for (y = j; y < (j + BLOCK_SIZE); y++) {
                             double soma = 0.0;
                             for (z = k; z < (k + BLOCK_SIZE); z++) {
                                 soma +=  A.dados[x * C.n + z] * B.dados[z * C.n + y];
                             }
                             C.dados[x * C.n + y] = soma;
                         }
                     }
                 }
             }
         }
     }

}

// ################### FUNCOES DEVICE E KERNEL ############################

// Funcao Kernel que executa na GPU. Cada thread calcula um elemento de C acumulando
// o resultado em ctmp. Metodo que utiliza muito a memoria global e por isso nao e mt eficiente
__global__ void mat_mult_kernel (MATRIX dA, MATRIX dB, MATRIX dC){
    double ctmp = 0.0;
    int i;
    int lin = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //if(lin > dA.m || col > dB.n) return;
    for (i=0; i<dA.n; ++i){
        ctmp += dA.dados[lin * dA.n + i] * dB.dados[i * dB.n + col];
    }
    dC.dados[lin * dC.n + col] = ctmp;
}

// Iniciando parte de multiplicação com ladrilho
__device__ double get_elemento (MATRIX mat, int i, int j){
    return mat.dados[i * mat.des + j];
}

__device__ void set_elemento (MATRIX mat, int i, int j, double val){
    mat.dados[i * mat.des + j] = val;
}

__device__ MATRIX get_sub_mat (MATRIX mat, int i, int j){
    MATRIX mat_sub;
    mat_sub.n = BLOCK_SIZE;
    mat_sub.m = BLOCK_SIZE;
    mat_sub.des = mat.des;
    mat_sub.dados = &mat.dados[mat.des * BLOCK_SIZE * i + BLOCK_SIZE * j];
    return mat_sub;
}

// Kernel para multiplicacao na GPU ladrilhada
__global__ void mat_mult_kernel_ladrilhada (MATRIX dA, MATRIX dB, MATRIX dC){
    // pegando a linha e coluna do bloco
    int bloco_lin = blockIdx.y;
    int bloco_col = blockIdx.x;

    // Cada bloco de threads calcula uma submatriz de C
    MATRIX C_sub = get_sub_mat (dC, bloco_lin, bloco_col);

    // Cada thread calcula um elemento de C_sub e acumula o resultado em ctmp
    double ctmp = 0;

    // Threads que estão dentro de C_sub
    int lin = threadIdx.y;
    int col = threadIdx.x;
    int j;

    // Multiplicar cada sub matriz para computar o valor de ctmp. Mesma ideia da
    // forma sem ladrilho
    for (j=0; j < (dA.n / BLOCK_SIZE); ++j){
        // subamtrizes do ladrilho
        MATRIX A_sub = get_sub_mat (dA, bloco_lin, j);
        MATRIX B_sub = get_sub_mat (dB, j, bloco_col);

        // sub matrizes que vao para memoria compartilhada
        __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Carregar as submatrizes da memorica do device para memoria compartilhada
        // Cada thread carrega um elemento
        As[lin][col] = get_elemento (A_sub, lin, col);
        Bs[lin][col] = get_elemento (B_sub, lin, col);

        // Sincornizar para ter certeza que foram carregadas
        __syncthreads();

        // Multiplicacao das submatrizes
        #pragma unroll
        for (int k=0; k < BLOCK_SIZE; ++k){
            ctmp += As[lin][k] * Bs[k][col];
        }

        // Novamente, outra sincronizacao
        __syncthreads();
    }

    // Seta o elemento computado por essas submatrizes.
    // Cada thread escreve um elemento
    set_elemento (C_sub, lin, col, ctmp);
}


// Parâmetros: m de A, n de A, e n de B
// Assumindo sempre que m de B = m de A
int main (int argc, char* argv[]){
    // Declarando e alocando as matrizes no host
    MATRIX A, B, C; // C = A dot B
    int gpu = 1; //  0 = Falso, 1 = True
    int nIter = 100; //Numero de iteracoes do kernel ou cpu
    // Alocando A
    A.m = atoi (argv[1]);
    A.n = atoi (argv[2]);
    A.dados = (double*)malloc(A.m * A.n * sizeof(double));
    if (A.dados == NULL){
        printf("Erro na alocacao da matriz A\n");
        exit(1);
    }

    // Alocando B
    B.m = A.n;
    B.n = atoi (argv[3]);
    B.dados = (double*)malloc(B.m * B.n * sizeof(double));
    if (B.dados == NULL){
        printf("Erro na alocacao da matriz B\n");
        exit(1);
    }

    // Alocando C
    C.m = A.m;
    C.n = B.n;
    C.dados = (double*)malloc(C.m * C.n * sizeof(double));
    if (C.dados == NULL){
        printf("Erro na alocacao da matriz C\n");
        exit(1);
    }

    // Preenchendo as matrizes
    srand(time(NULL));
    preencher_mat (A, 'r');
    preencher_mat (B, 'r');

    //print_mat (A, -1);
    //print_mat (B, -1);

    if (gpu == 0){
        clock_t t_ini, t_fim;
        double t_final;
        int i;
        t_ini = clock();

        #pragma omp paralell for shared (nIter, A, B, C) private(i)
        for (i=0; i < nIter; i++){
            mult_mat_cpu_ladrilhada (A, B, C);
        }
        t_fim = clock();

        t_final = (((double)(t_fim - t_ini)) / CLOCKS_PER_SEC)*1000; // tempo em mseg

        float mseg_por_matriz = t_final/nIter;
        double flops_por_matriz = 2.0 * (double)A.n * (double)A.m * (double)B.n;
        double giga_flops = (flops_por_matriz * 1.0e-9f) / (mseg_por_matriz / 1000.0f);
        // Performance em GFLOPS/s e Tempo em mseg
        printf(
            "%.2f , %.3f \n",
            giga_flops,
            mseg_por_matriz);

    }else if (gpu==1){
        // Alocando e enviando as matrizes para GPU
        MATRIX dA, dB, dC;
        cudaError_t e;

        // Alocando A
        dA.m = A.m;
        dA.n = dA.des = A.n;
        e = cudaMalloc (&dA.dados, dA.m * dA.n * sizeof(double));
        if (e != cudaSuccess){
            printf("Erro no CUDA malloc A: %s\n",cudaGetErrorString(e));
            exit(1);
        }
        // Copiando A
        e = cudaMemcpy (dA.dados, A.dados, dA.m * dA.n * sizeof(double), cudaMemcpyHostToDevice);
        if (e != cudaSuccess){
            printf("Erro ao copiar matriz A para GPU: %s\n",cudaGetErrorString(e));
            exit(1);
        }

        // Alocando B
        dB.m = B.m;
        dB.n = dB.des = B.n;
        e = cudaMalloc (&dB.dados, dB.m * dB.n * sizeof(double));
        if (e != cudaSuccess){
            printf("Erro no CUDA malloc B: %s\n",cudaGetErrorString(e));
            exit(1);
        }
        // Copiando A
        e = cudaMemcpy (dB.dados, B.dados, dB.m * dB.n * sizeof(double), cudaMemcpyHostToDevice);
        if (e != cudaSuccess){
            printf("Erro ao copiar matriz B para GPU: %s\n",cudaGetErrorString(e));
            exit(1);
        }

        // Alocando C
        dC.m = C.m;
        dC.n = dC.des = C.n;
        e = cudaMalloc (&dC.dados, dC.m * dC.n * sizeof(double));
        if (e != cudaSuccess){
            printf("Erro no CUDA malloc C: %s\n",cudaGetErrorString(e));
            exit(1);
        }

        // Criando eventos para pegar informacoes da GPU
        cudaEvent_t inicio;
        e = cudaEventCreate(&inicio);
        if (e != cudaSuccess){
            fprintf(stderr, "Erro ao criar o evento inicio: %s\n", cudaGetErrorString(e));
            exit(1);
        }

        cudaEvent_t fim;
        e = cudaEventCreate(&fim);
        if (e != cudaSuccess){
            fprintf(stderr, "Erro ao criar evento fim: %s\n", cudaGetErrorString(e));
            exit(1);
        }

        // Gravando o evento inicio
        e = cudaEventRecord(inicio, NULL);
        if (e != cudaSuccess){
            fprintf(stderr, "Erro ao gravar evento inicio: %s\n", cudaGetErrorString(e));
            exit(1);
        }

        // Definindo o tamanho de grid, bloco e threads
        dim3 dim_bloco (BLOCK_SIZE, BLOCK_SIZE);
        dim3 dim_grid (dB.n/dim_bloco.x, dA.m/dim_bloco.y);

        //#################### EXECUTANDO KERNEL ###############################
        // Executando o Kernel nIter vezes
        for (int j = 0; j < nIter; j++){
            mat_mult_kernel_ladrilhada<<<dim_grid, dim_bloco>>>(dA,dB,dC);
        }
        //#################### EXECUTANDO KERNEL ###############################

        // Gravando evento fim
        e = cudaEventRecord(fim, NULL);
        if (e != cudaSuccess){
            fprintf(stderr, "Erro ao gravar evento fim: %s\n", cudaGetErrorString(e));
            exit(1);
        }

        // Esperando evento encerrar
        e = cudaEventSynchronize(fim);
        if (e != cudaSuccess){
            fprintf(stderr, "Erro ao sincronizar evento fim: %s\n", cudaGetErrorString(e));
            exit(1);
        }

        float mseg_total = 0.0f;
        // Pegando tempo gasto
        e = cudaEventElapsedTime(&mseg_total, inicio, fim);
        if (e != cudaSuccess){
            fprintf(stderr, "Erro ao pegar tempo gasto entre os eventos: %s\n", cudaGetErrorString(e));
            exit(1);
        }

        // Calculando e imprimindo a performance
        float mseg_por_matriz = mseg_total/nIter;
        double flops_por_matriz = 2.0 * (double)A.n * (double)A.m * (double)B.n;
        double giga_flops = (flops_por_matriz * 1.0e-9f) / (mseg_por_matriz / 1000.0f);
        printf(
            "%.2f, %.3f\n",
            giga_flops,
            mseg_por_matriz);

        // Copiando resultado da GPU pra host
        e = cudaMemcpy (C.dados, dC.dados, dC.m * dC.n * sizeof(double), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess){
            printf("Erro ao copiar matrix C para CPU: %s\n",cudaGetErrorString(e));
            exit(1);
        }

        // Liberando a memoria tanto do device
        cudaFree(dA.dados); cudaFree(dB.dados); cudaFree(dC.dados);
    }

    // Imprimindo o resultado da multiplicacao
    //print_mat (C, -1);

    // Liberando a memoria tanto do host
    free(A.dados); free(B.dados); free(C.dados);
    return 0;
}
