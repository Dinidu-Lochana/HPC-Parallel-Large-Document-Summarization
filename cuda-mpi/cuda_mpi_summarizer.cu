#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>

#define CHUNK_SIZE 2000
#define MAX_CHUNKS 1000
#define MAX_SUMMARY_SIZE 2048
#define MAX_FILENAME_SIZE 256

typedef struct {
    char text[CHUNK_SIZE + 100];
    int chunk_id;
    int text_length;
} DocumentChunk;

typedef struct {
    char summary[MAX_SUMMARY_SIZE];
    int chunk_id;
    int summary_length;
} ChunkSummary;

// Function declarations
void master_process(int num_procs, char *input_file, char *topic);
void worker_process(int rank, char *topic);
int split_document_cuda(char *filename, DocumentChunk **out_chunks);
void call_python_summarizer(char *chunk_text, char *topic, char *output_summary, int chunk_id);
void combine_summaries(ChunkSummary *summaries, int num_summaries, char *final_summary, char *topic);

// ==========================================
// 1. CUDA Kernel for Chunk Boundary Parsing
// ==========================================
__global__ void find_chunk_boundaries(const char *text, int *chunk_boundaries, int text_length, int max_chunk_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < MAX_CHUNKS) {
        int start_pos = idx * max_chunk_size;
        if (start_pos >= text_length) {
            chunk_boundaries[idx] = -1;
            return;
        }
        int end_pos = start_pos + max_chunk_size;
        if (end_pos >= text_length) {
            chunk_boundaries[idx] = text_length;
            return;
        }
        int current_pos = end_pos;
        // Backtrack to avoid cutting words
        while (current_pos > start_pos && text[current_pos] != ' ' && text[current_pos] != '\n') {
            current_pos--;
        }
        if (current_pos == start_pos) {
            chunk_boundaries[idx] = end_pos;
        } else {
            chunk_boundaries[idx] = current_pos;
        }
    }
}

// ==========================================
// 2. Main Execution (MPI Entry Point)
// ==========================================
int main(int argc, char *argv[]) {
    int rank, num_procs;
    char input_file[MAX_FILENAME_SIZE];
    char topic[256] = "General";
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    if (argc < 2) {
        if (rank == 0) {
            printf("Usage: %s <input_file> [topic]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    strcpy(input_file, argv[1]);
    if (argc >= 3) {
        strcpy(topic, argv[2]);
    }
    
    if (rank == 0) {
        printf("=================================================\n");
        printf("Hybrid CUDA + MPI Document Summarization\n");
        printf("=================================================\n");
        printf("Nodes/Processes: %d\n", num_procs);
        master_process(num_procs, input_file, topic);
    } else {
        worker_process(rank, topic);
    }
    
    MPI_Finalize();
    return 0;
}

// ==========================================
// 3. Master Process (Handles CUDA & Distribution)
// ==========================================
void master_process(int num_procs, char *input_file, char *topic) {
    DocumentChunk *chunks = NULL;
    ChunkSummary *summaries = NULL;
    int num_chunks;
    double start_time, end_time;
    MPI_Status status;
    
    start_time = MPI_Wtime();
    
    // Step 1: Use CUDA to parse the file and find boundaries
    num_chunks = split_document_cuda(input_file, &chunks);
    if (num_chunks <= 0) {
        printf("Master: Failed to split document.\n");
        int term = -1;
        for(int i = 1; i < num_procs; i++) MPI_Send(&term, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        return;
    }
    
    printf("Master: Successfully split document into %d chunks using CUDA GPU.\n", num_chunks);
    
    summaries = (ChunkSummary *)malloc(num_chunks * sizeof(ChunkSummary));
    int chunks_sent = 0;
    int chunks_received = 0;
    
    // Initial distribution to MPI workers
    for (int i = 1; i < num_procs && chunks_sent < num_chunks; i++) {
        MPI_Send(&chunks[chunks_sent].chunk_id, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        MPI_Send(&chunks[chunks_sent].text_length, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        MPI_Send(chunks[chunks_sent].text, chunks[chunks_sent].text_length, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        chunks_sent++;
    }
    
    // Receive summaries and send remaining chunks (Round Robin / Worker pool)
    while (chunks_received < num_chunks) {
        int chunk_id, summary_length;
        MPI_Recv(&chunk_id, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&summary_length, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(summaries[chunks_received].summary, summary_length, MPI_CHAR, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &status);
        
        summaries[chunks_received].chunk_id = chunk_id;
        summaries[chunks_received].summary_length = summary_length;
        summaries[chunks_received].summary[summary_length] = '\0';
        
        chunks_received++;
        
        if (chunks_sent < num_chunks) {
            MPI_Send(&chunks[chunks_sent].chunk_id, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            MPI_Send(&chunks[chunks_sent].text_length, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            MPI_Send(chunks[chunks_sent].text, chunks[chunks_sent].text_length, MPI_CHAR, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            chunks_sent++;
        } else {
            int term = -1;
            MPI_Send(&term, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
        }
    }
    
    printf("\nAll %d chunks processed by MPI Workers. Master is combining summaries...\n", num_chunks);
    char final_summary[MAX_SUMMARY_SIZE * 10];
    combine_summaries(summaries, num_chunks, final_summary, topic);
    
    end_time = MPI_Wtime();
    printf("\n=================================================\n");
    printf("FINAL SUMMARY\n");
    printf("=================================================\n");
    printf("%s\n", final_summary);
    printf("=================================================\n");
    printf("Hybrid (CUDA + MPI) Execution Time: %.4f seconds\n", end_time - start_time);
    
    free(chunks);
    free(summaries);
}

// ==========================================
// 4. Worker Process (API requests)
// ==========================================
void worker_process(int rank, char *topic) {
    MPI_Status status;
    int chunk_id, text_length;
    char chunk_text[CHUNK_SIZE + 100];
    char summary[MAX_SUMMARY_SIZE];
    
    while (1) {
        // Receive chunk ID
        MPI_Recv(&chunk_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        if (chunk_id == -1) {
            break; // Termination signal from master
        }
        
        // Receive text length and actual text
        MPI_Recv(&text_length, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(chunk_text, text_length, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        chunk_text[text_length] = '\0';
        
        // Let python LLM API summarize the chunk
        call_python_summarizer(chunk_text, topic, summary, chunk_id);
        
        // Send back the results to Master
        int summary_length = strlen(summary);
        MPI_Send(&chunk_id, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&summary_length, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(summary, summary_length, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
    }
}

// ==========================================
// 5. CUDA Subroutine Helper
// ==========================================
int split_document_cuda(char *filename, DocumentChunk **out_chunks) {
    int is_pdf = 0;
    char *ext = strrchr(filename, '.');
    if (ext && strcmp(ext, ".pdf") == 0) is_pdf = 1;
    
    char *full_text = NULL;
    int file_size = 0;
    
    if (is_pdf) {
        char temp_txt_file[256];
        sprintf(temp_txt_file, "temp_hybrid_pdf_extract_%d.txt", getpid());
        char command[1024];
        sprintf(command, "python.exe ../python_llm/summarizer.py extract_pdf \"%s\" \"%s\"", filename, temp_txt_file);
        if (system(command) != 0) return -1;
        
        FILE *file = fopen(temp_txt_file, "r");
        if (!file) return -1;
        fseek(file, 0, SEEK_END);
        file_size = ftell(file);
        fseek(file, 0, SEEK_SET);
        full_text = (char *)malloc(file_size + 1);
        fread(full_text, 1, file_size, file);
        full_text[file_size] = '\0';
        fclose(file);
        remove(temp_txt_file);
    } else {
        FILE *file = fopen(filename, "r");
        if (!file) return -1;
        fseek(file, 0, SEEK_END);
        file_size = ftell(file);
        fseek(file, 0, SEEK_SET);
        full_text = (char *)malloc(file_size + 1);
        fread(full_text, 1, file_size, file);
        full_text[file_size] = '\0';
        fclose(file);
    }
    
    char *device_text;
    int *device_boundaries;
    int *host_boundaries = (int *)malloc(MAX_CHUNKS * sizeof(int));
    
    cudaMalloc((void **)&device_text, file_size + 1);
    cudaMalloc((void **)&device_boundaries, MAX_CHUNKS * sizeof(int));
    cudaMemcpy(device_text, full_text, file_size + 1, cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (MAX_CHUNKS + threadsPerBlock - 1) / threadsPerBlock;
    
    // Execute CUDA Kernel
    find_chunk_boundaries<<<blocksPerGrid, threadsPerBlock>>>(device_text, device_boundaries, file_size, CHUNK_SIZE);
    
    cudaMemcpy(host_boundaries, device_boundaries, MAX_CHUNKS * sizeof(int), cudaMemcpyDeviceToHost);
    
    DocumentChunk *chunks = (DocumentChunk *)malloc(MAX_CHUNKS * sizeof(DocumentChunk));
    int current_start = 0;
    int num_chunks = 0;
    
    for (int i = 0; i < MAX_CHUNKS; i++) {
        if (current_start >= file_size) break;
        int end_pos = host_boundaries[i];
        if (end_pos == -1) break;
        
        if (i > 0) {
            end_pos = current_start + CHUNK_SIZE;
            if (end_pos > file_size) end_pos = file_size;
            int temp = end_pos;
            while (temp > current_start && full_text[temp] != ' ' && full_text[temp] != '\n') temp--;
            if (temp != current_start) end_pos = temp;
        }
        
        int chunk_length = end_pos - current_start;
        chunks[num_chunks].chunk_id = num_chunks;
        strncpy(chunks[num_chunks].text, full_text + current_start, chunk_length);
        chunks[num_chunks].text[chunk_length] = '\0';
        chunks[num_chunks].text_length = chunk_length;
        
        current_start = end_pos;
        num_chunks++;
    }
    
    free(full_text);
    free(host_boundaries);
    cudaFree(device_text);
    cudaFree(device_boundaries);
    
    *out_chunks = chunks;
    return num_chunks;
}

// ==========================================
// 6. Python Integration Helpers
// ==========================================
void call_python_summarizer(char *chunk_text, char *topic, char *output_summary, int chunk_id) {
    char temp_input_file[256];
    char temp_output_file[256];
    
    int pid = getpid();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    sprintf(temp_input_file, "temp_hybrid_chunk_%d_%d.txt", rank, chunk_id);
    sprintf(temp_output_file, "temp_hybrid_summary_%d_%d.txt", rank, chunk_id);
    
    FILE *input = fopen(temp_input_file, "w");
    if (input) {
        fprintf(input, "%s", chunk_text);
        fclose(input);
    }
    
    char command[1024];
    sprintf(command, "python.exe ../python_llm/summarizer.py summarize_chunk \"%s\" \"%s\" \"%s\"", 
            temp_input_file, topic, temp_output_file);
            
    time_t rawtime_start, rawtime_end;
    char time_buffer_start[80], time_buffer_end[80];
    time(&rawtime_start);
    strftime(time_buffer_start, sizeof(time_buffer_start), "%I:%M:%S %p", localtime(&rawtime_start));
    
    struct timespec start_ts, end_ts;
    clock_gettime(CLOCK_MONOTONIC, &start_ts);
    
    printf("[Hybrid Worker %d] Chunk %d START time: %s\n", rank, chunk_id, time_buffer_start);
    
    system(command);
    
    time(&rawtime_end);
    strftime(time_buffer_end, sizeof(time_buffer_end), "%I:%M:%S %p", localtime(&rawtime_end));
    clock_gettime(CLOCK_MONOTONIC, &end_ts);
    double elapsed = (end_ts.tv_sec - start_ts.tv_sec) + (end_ts.tv_nsec - start_ts.tv_nsec) / 1e9;
    
    printf("[Hybrid Worker %d] Chunk %d END time: %s (Execution: %.2f sec)\n", rank, chunk_id, time_buffer_end, elapsed);
    
    FILE *output = fopen(temp_output_file, "r");
    if (output) {
        fread(output_summary, 1, MAX_SUMMARY_SIZE - 1, output);
        output_summary[MAX_SUMMARY_SIZE - 1] = '\0';
        fclose(output);
    } else {
        strcpy(output_summary, "[Summary generation failed]");
    }
    
    remove(temp_input_file);
    remove(temp_output_file);
}

void combine_summaries(ChunkSummary *summaries, int num_summaries, char *final_summary, char *topic) {
    char combined_file[256];
    sprintf(combined_file, "temp_hybrid_combined_%d.txt", getpid());
    
    FILE *combined = fopen(combined_file, "w");
    if (combined) {
        for (int i = 0; i < num_summaries; i++) {
            fprintf(combined, "%s\n\n", summaries[i].summary);
        }
        fclose(combined);
    }
    
    char output_file[256];
    sprintf(output_file, "temp_hybrid_final_%d.txt", getpid());
    
    char command[1024];
    sprintf(command, "python.exe ../python_llm/summarizer.py combine_summaries \"%s\" \"%s\" \"%s\"", 
            combined_file, topic, output_file);
    
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
