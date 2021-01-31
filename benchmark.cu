#include <iostream>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// USE_HALF and USE_FLOAT are mutually exclusive. Only one can be used at a time.
// #define USE_HALF
#define USE_FLOAT

#define EASY_COPY
#define NUM_ITERATIONS 20

void fill_random(float* arr, int dim_size) {
    for(int i = 0; i < dim_size * dim_size; i++) {
        #if defined(USE_FLOAT)
            arr[i] = (float)rand()/RAND_MAX;
        #elif defined(USE_HALF)
            arr[i] = __float2half(rand()/RAND_MAX);
        #endif
    }
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);
    std::chrono::high_resolution_clock::time_point cpu_start, cpu_end;
    std::chrono::duration<double> cpu_duration;
    std::cout << std::fixed;

    // Create handle for cublas
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Create variables for measuring execution time
    cudaEvent_t gpu_start, gpu_stop;
    cudaEventCreate(&gpu_start);
    cudaEventCreate(&gpu_stop);

    // The max dimension of the matrix
    // eg if max_dim = 16, the max matrix size will be 16x16
    int min_dim = 16384;
    int max_dim = 16384;
    // int max_dim = 1024*16;

    // 128, 256, 512, 1024, 2048, 4096, 8192, 16384



    // Allocate matrices on the host
    std::cout << "Allocating memory on host... ";
    cpu_start = std::chrono::high_resolution_clock::now();
    float* h_A = (float *)malloc(max_dim * max_dim * sizeof(float));
    float* h_B = (float *)malloc(max_dim * max_dim * sizeof(float));
    float* h_C = (float *)malloc(max_dim * max_dim * sizeof(float));
    cpu_end = std::chrono::high_resolution_clock::now();
    cpu_duration = std::chrono::duration_cast<std::chrono::duration<double>>(cpu_end - cpu_start);
    std::cout << "done (" << cpu_duration.count() << " seconds)" << std::endl;


    // Fill matrices with random data
    std::cout << "Filling memory with random data... ";
    cpu_start = std::chrono::high_resolution_clock::now();
    fill_random(h_A, max_dim);
    fill_random(h_B, max_dim);
    cpu_end = std::chrono::high_resolution_clock::now();
    cpu_duration = std::chrono::duration_cast<std::chrono::duration<double>>(cpu_end - cpu_start);
    std::cout << "done (" << cpu_duration.count() << " seconds)" << std::endl;


    // Create managed memory on the GPU
    std::cout << "Allocating memory on device... ";
    cpu_start = std::chrono::high_resolution_clock::now();

    #if defined(USE_FLOAT)
        float *d_A, *d_B, *d_C;
        cudaMallocManaged(&d_A, max_dim * max_dim * sizeof(float));
        cudaMallocManaged(&d_B, max_dim * max_dim * sizeof(float));
        cudaMallocManaged(&d_C, max_dim * max_dim * sizeof(float));
    #elif defined(USE_HALF)
        __half *d_A, *d_B, *d_C;
        cudaMallocManaged(&d_A, max_dim * max_dim * sizeof(__half));
        cudaMallocManaged(&d_B, max_dim * max_dim * sizeof(__half));
        cudaMallocManaged(&d_C, max_dim * max_dim * sizeof(__half));
    #endif

    cpu_end = std::chrono::high_resolution_clock::now();
    cpu_duration = std::chrono::duration_cast<std::chrono::duration<double>>(cpu_end - cpu_start);
    std::cout << "done (" << cpu_duration.count() << " seconds)" << std::endl;


    // Copy data from host arrays to device arrays
    std::cout << "Copying data from host to device... ";
    cpu_start = std::chrono::high_resolution_clock::now();

    #if defined(USE_FLOAT)
        cudaMemcpy(d_A, h_A, max_dim * max_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, max_dim * max_dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, h_C, max_dim * max_dim * sizeof(float), cudaMemcpyHostToDevice);
    #elif defined(USE_HALF)
        cudaMemcpy(d_A, h_A, max_dim * max_dim * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, max_dim * max_dim * sizeof(__half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_C, h_C, max_dim * max_dim * sizeof(__half), cudaMemcpyHostToDevice);
    #endif

    cpu_end = std::chrono::high_resolution_clock::now();
    cpu_duration = std::chrono::duration_cast<std::chrono::duration<double>>(cpu_end - cpu_start);
    std::cout << "done (" << cpu_duration.count() << " seconds)" << std::endl;


    // Run benchmark
    std::cout << "Starting GPU benchmark... " << std::endl;

    #if defined(USE_FLOAT)
        float alpha = 1;
        float beta = 1;
    #elif defined(USE_HALF)
        __half alpha = 1.0;
        __half beta = 0.0;
    #endif

    float time_ms, final_time;
    double num_flop, final_flops;
    for(int i = min_dim; i <= max_dim; i*=2) {
        #if defined(EASY_COPY)
            std::cout << i << "\t";
        #else
            std::cout << "Running with size " << i << "... ";
        #endif
        cudaEventRecord(gpu_start);

        for (int ii = 0; ii < NUM_ITERATIONS; ii++) {
            #if defined(USE_FLOAT)
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, i, i, i, &alpha, d_A, i, d_B, i, &beta, d_C, i);
            #elif defined(USE_HALF)
                cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, i, i, i, &alpha, d_A, i, d_B, i, &beta, d_C, i);
            #endif
        }

        cudaEventRecord(gpu_stop);
        cudaEventSynchronize(gpu_stop);
        cudaEventElapsedTime(&time_ms, gpu_start, gpu_stop);

        // NOTE: I am not 100% certain on this formula.
        // https://math.stackexchange.com/a/2200792 has the math for AB, and I attempted to adapt it for aAB + bC
        // i^2 * (3i + 1)
        // num_flop = (unsigned long long)(i * i) * ((unsigned long long)(3 * i) + 1);
        num_flop = (unsigned long long)(i * i) * ((unsigned long long)(2 * i) - 1);

        final_time = ((time_ms / 1000.0) / NUM_ITERATIONS);
        final_flops = (num_flop / (double)final_time);

        #if defined(EASY_COPY)
            std::cout << final_time << "\t" << final_flops;
        #else
            std::cout << "done (avg " << final_time << " seconds, ";
            if (final_flops > 1000000000000.0) { // Tera Flops
                printf("%.2f TFlops)", (final_flops / 1000000000000.0));
                // std::cout << (final_flops / 1000000000000.0) << " TFlops)";
            } else if (final_flops > 1000000000.0) { // Giga Flops
                printf("%.2f GFlops)", (final_flops / 1000000000.0));
                // std::cout << (final_flops / 1000000000.0) << " GFlops)";
            } else if (final_flops > 1000000.0) { // Mega Flops
                printf("%.2f MFlops)", (final_flops / 1000000.0));
                // std::cout << (final_flops / 1000000.0) << " MFlops)";
            } else if (final_flops > 1000.0) { // Kila Flops
                printf("%.2f KFlops)", (final_flops / 1000.0));
                // std::cout << (final_flops / 1000.0) << " KFlops)";
            } else {
                printf("%.2f Flops)", final_flops);
                // std::cout << final_flops << " Flops)";
            }
        #endif


        std::cout << std::endl;
    }
    // (rowA * colB)(rowA + colB - 1) + (rowA + rowB)


    // Free memory on host and device
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}
