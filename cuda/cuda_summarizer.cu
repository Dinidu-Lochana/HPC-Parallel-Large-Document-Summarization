#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>

#define CHUNK_SIZE 2000
#define MAX_CHUNKS 1000
#define MAX_SUMMARY_SIZE 2048

// Structure to hold chunk data for pthread
typedef struct {
    int chunk_id;
    char *text;
    int text_length;
    char *topic;
    char summary[MAX_SUMMARY_SIZE];
} ChunkData;

// CUDA Kernel to find valid split points
__global__ void find_chunk_boundaries(const char *text, int *chunk_boundaries, int text_length, int max_chunk_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread calculates the boundary for chunk 'idx'
    if (idx < MAX_CHUNKS) {
        int start_pos = idx * max_chunk_size;
        
        if (start_pos >= text_length) {
            chunk_boundaries[idx] = -1; // End of text
            return;
        }
        
        int end_pos = start_pos + max_chunk_size;
        if (end_pos >= text_length) {
            chunk_boundaries[idx] = text_length;
            return;
        }
        
        // Find the nearest space or newline backwards
        int current_pos = end_pos;
        while (current_pos > start_pos && text[current_pos] != ' ' && text[current_pos] != '\n') {
            current_pos--;
        }
        
        // If no space found, force split at max_chunk_size
        if (current_pos == start_pos) {
            chunk_boundaries[idx] = end_pos;
        } else {
            chunk_boundaries[idx] = current_pos;
        }
    }
}

// Function executed by each thread to call the Python API for summarization
void *summarize_chunk_thread(void *arg) {
    ChunkData *data = (ChunkData *)arg;
    
    char temp_input_file[256];
    char temp_output_file[256];
    
    // Unique temporary files based on pthread ID or chunk ID
    sprintf(temp_input_file, "temp_cuda_chunk_%d.txt", data->chunk_id);
    sprintf(temp_output_file, "temp_cuda_summary_%d.txt", data->chunk_id);
    
    // Write chunk text to temp file
    FILE *input = fopen(temp_input_file, "w");
    if (input) {
        fprintf(input, "%s", data->text);
        fclose(input);
    } else {
        strcpy(data->summary, "[Error creating temp file]");
        return NULL;
    }
    
    // Construct command to call python_llm/summarizer.py
    char command[1024];
    sprintf(command, "python.exe ../python_llm/summarizer.py summarize_chunk \"%s\" \"%s\" \"%s\"", 
            temp_input_file, data->topic, temp_output_file);
    
    time_t rawtime_start, rawtime_end;
    char time_buffer_start[80], time_buffer_end[80];

    time(&rawtime_start);
    strftime(time_buffer_start, sizeof(time_buffer_start), "%I:%M:%S %p", localtime(&rawtime_start));
    
    struct timespec start_ts, end_ts;
    clock_gettime(CLOCK_MONOTONIC, &start_ts);

    printf("\n[Cuda Summarizer Thread %d] Chunk %d START time: %s\n", data->chunk_id, data->chunk_id, time_buffer_start);
    
    int result = system(command);
    
    time(&rawtime_end);
    strftime(time_buffer_end, sizeof(time_buffer_end), "%I:%M:%S %p", localtime(&rawtime_end));
    
    clock_gettime(CLOCK_MONOTONIC, &end_ts);
    double elapsed = (end_ts.tv_sec - start_ts.tv_sec) + (end_ts.tv_nsec - start_ts.tv_nsec) / 1e9;
    
    printf("[Cuda Summarizer Thread %d] Chunk %d END time: %s (Execution time: %.2f seconds)\n", data->chunk_id, data->chunk_id, time_buffer_end, elapsed);
    
    // Read summary
    FILE *output = fopen(temp_output_file, "r");
    if (output) {
        fread(data->summary, 1, MAX_SUMMARY_SIZE - 1, output);
        data->summary[MAX_SUMMARY_SIZE - 1] = '\0';
        fclose(output);
    } else {
        strcpy(data->summary, "[Summary generation failed]");
    }
    
    // Cleanup temp files
    remove(temp_input_file);
    remove(temp_output_file);
    
    return NULL;
}

// Combine summaries using Python LLM
void combine_summaries(ChunkData *chunks, int num_chunks, char *final_summary, const char *topic) {
    char combined_file[256] = "temp_cuda_combined.txt";
    
    FILE *combined = fopen(combined_file, "w");
    if (combined) {
        for (int i = 0; i < num_chunks; i++) {
            fprintf(combined, "%s\n\n", chunks[i].summary);
        }
        fclose(combined);
    }
    
    char output_file[256] = "temp_cuda_final.txt";
    char command[1024];
    sprintf(command, "python.exe ../python_llm/summarizer.py combine_summaries \"%s\" \"%s\" \"%s\"", 
            combined_file, topic, output_file);
    time_t rawtime;
    struct tm * timeinfo;
    char time_buffer[80];
    time(&rawtime);
    timeinfo = localtime(&rawtime);
    strftime(time_buffer, sizeof(time_buffer), "%I:%M:%S %p", timeinfo);

    printf("\n[Cuda Summarizer Final] Combining summaries started at %s\n", time_buffer);
    
    system(command);
    
    FILE *output = fopen(output_file, "r");
    if (output) {
        fread(final_summary, 1, MAX_SUMMARY_SIZE * 10 - 1, output);
        final_summary[MAX_SUMMARY_SIZE * 10 - 1] = '\0';
        fclose(output);
    } else {
        strcpy(final_summary, "[Final summary generation failed]");
    }
    
    remove(combined_file);
    remove(output_file);
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <input_file> [topic]\n", argv[0]);
        return 1;
    }
    
    char *input_file = argv[1];
    char topic[256] = "General";
    if (argc >= 3) {
        strcpy(topic, argv[2]);
    }
    
    printf("=================================================\n");
    printf("CUDA Parallel Document Summarization\n");
    printf("=================================================\n");
    
    // Capture overall start time
    time_t prog_start_raw;
    char prog_start_buf[80];
    struct timespec prog_start_ts;
    time(&prog_start_raw);
    strftime(prog_start_buf, sizeof(prog_start_buf), "%I:%M:%S %p", localtime(&prog_start_raw));
    clock_gettime(CLOCK_MONOTONIC, &prog_start_ts);
    printf("[Cuda Summarizer] Execution started at: %s\n", prog_start_buf);
    printf("=================================================\n");
    
    // Read file
    FILE *file = fopen(input_file, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", input_file);
        return 1;
    }
    
    fseek(file, 0, SEEK_END);
    int file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char *host_text = (char *)malloc(file_size + 1);
    fread(host_text, 1, file_size, file);
    host_text[file_size] = '\0';
    fclose(file);
    
    // Allocate device memory
    char *device_text;
    int *device_boundaries;
    int *host_boundaries = (int *)malloc(MAX_CHUNKS * sizeof(int));
    
    cudaMalloc((void **)&device_text, file_size + 1);
    cudaMalloc((void **)&device_boundaries, MAX_CHUNKS * sizeof(int));
    
    cudaMemcpy(device_text, host_text, file_size + 1, cudaMemcpyHostToDevice);
    
    // Launch CUDA Kernel to find chunk boundaries
    int threadsPerBlock = 256;
    int blocksPerGrid = (MAX_CHUNKS + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching CUDA kernel to calculate chunk boundaries...\n");
    find_chunk_boundaries<<<blocksPerGrid, threadsPerBlock>>>(device_text, device_boundaries, file_size, CHUNK_SIZE);
    
    cudaMemcpy(host_boundaries, device_boundaries, MAX_CHUNKS * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Process boundaries to create chunks
    int num_chunks = 0;
    int current_start = 0;
    
    // Heap-allocate chunk_data to avoid stack overflow on large documents
    ChunkData *chunk_data = (ChunkData *)malloc(MAX_CHUNKS * sizeof(ChunkData));
    
    for (int i = 0; i < MAX_CHUNKS; i++) {
        if (current_start >= file_size) break;
        
        // Use the GPU-computed word-safe boundary directly
        int end_pos = host_boundaries[i];
        if (end_pos == -1) break;
        if (end_pos > file_size) end_pos = file_size;
        
        int chunk_length = end_pos - current_start;
        if (chunk_length <= 0) break;
        
        chunk_data[num_chunks].chunk_id = num_chunks;
        chunk_data[num_chunks].text = (char *)malloc(chunk_length + 1);
        strncpy(chunk_data[num_chunks].text, host_text + current_start, chunk_length);
        chunk_data[num_chunks].text[chunk_length] = '\0';
        chunk_data[num_chunks].text_length = chunk_length;
        chunk_data[num_chunks].topic = topic;
        
        current_start = end_pos;
        num_chunks++;
    }
    
    printf("Document split into %d chunks using CUDA.\n", num_chunks);
    printf("Processing chunks in parallel using pthreads...\n");
    
    // Heap-allocate threads array to avoid stack overflow on large documents
    pthread_t *threads = (pthread_t *)malloc(num_chunks * sizeof(pthread_t));
    for (int i = 0; i < num_chunks; i++) {
        pthread_create(&threads[i], NULL, summarize_chunk_thread, &chunk_data[i]);
    }
    
    // Wait for all threads to complete
    for (int i = 0; i < num_chunks; i++) {
        pthread_join(threads[i], NULL);
    }
    free(threads);
    
    printf("\nAll chunks processed. Combining summaries...\n");
    
    char final_summary[MAX_SUMMARY_SIZE * 10];
    combine_summaries(chunk_data, num_chunks, final_summary, topic);
    
    printf("\n=================================================\n");
    printf("FINAL SUMMARY\n");
    printf("=================================================\n");
    printf("%s\n", final_summary);
    printf("=================================================\n");
    
    // Print execution finished time and total elapsed
    time_t prog_end_raw;
    char prog_end_buf[80];
    struct timespec prog_end_ts;
    time(&prog_end_raw);
    strftime(prog_end_buf, sizeof(prog_end_buf), "%I:%M:%S %p", localtime(&prog_end_raw));
    clock_gettime(CLOCK_MONOTONIC, &prog_end_ts);
    double total_elapsed = (prog_end_ts.tv_sec - prog_start_ts.tv_sec) + (prog_end_ts.tv_nsec - prog_start_ts.tv_nsec) / 1e9;
    printf("[Cuda Summarizer] Execution finished at: %s\n", prog_end_buf);
    printf("[Cuda Summarizer] Total execution time: %.2f seconds\n", total_elapsed);
    printf("=================================================\n");
    
    // Cleanup
    free(host_text);
    free(host_boundaries);
    for (int i = 0; i < num_chunks; i++) {
        free(chunk_data[i].text);
    }
    free(chunk_data);
    cudaFree(device_text);
    cudaFree(device_boundaries);
    
    return 0;
}
