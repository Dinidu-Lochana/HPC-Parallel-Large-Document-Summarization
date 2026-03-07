#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_CHUNK_SIZE 4096
#define MAX_SUMMARY_SIZE 2048
#define MAX_FILENAME_SIZE 256

typedef struct {
    char text[MAX_CHUNK_SIZE];
    int chunk_id;
    int text_length;
} DocumentChunk;

typedef struct {
    char summary[MAX_SUMMARY_SIZE];
    int chunk_id;
    int summary_length;
} ChunkSummary;

void master_process(int num_procs, char *input_file, char *topic);
void worker_process(int rank, char *topic);
int read_document_chunks(char *filename, DocumentChunk **chunks);
void call_python_summarizer(char *chunk_text, char *topic, char *output_summary);
void combine_summaries(ChunkSummary *summaries, int num_summaries, char *final_summary, char *topic);

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
            printf("Example: mpirun -np 4 %s document.txt \"Machine Learning\"\n", argv[0]);
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
        printf("MPI Parallel Document Summarization\n");
        printf("=================================================\n");
        printf("Number of processes: %d\n", num_procs);
        printf("Input file: %s\n", input_file);
        printf("Topic: %s\n", topic);
        printf("=================================================\n\n");
        
        master_process(num_procs, input_file, topic);
    } else {
        worker_process(rank, topic);
    }
    
    MPI_Finalize();
    return 0;
}

void master_process(int num_procs, char *input_file, char *topic) {
    DocumentChunk *chunks = NULL;
    ChunkSummary *summaries = NULL;
    int num_chunks, i;
    double start_time, end_time;
    MPI_Status status;
    
    start_time = MPI_Wtime();
    
    num_chunks = read_document_chunks(input_file, &chunks);
    
    if (num_chunks <= 0) {
        printf("Error: Could not read document chunks\n");
        
        int terminate_signal = -1;
        for (i = 1; i < num_procs; i++) {
            MPI_Send(&terminate_signal, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        return;
    }
    
    printf("Document split into %d chunks\n", num_chunks);
    printf("Distributing chunks to %d worker processes...\n\n", num_procs - 1);
    
    summaries = (ChunkSummary *)malloc(num_chunks * sizeof(ChunkSummary));
    
    int chunks_sent = 0;
    int chunks_received = 0;
    
    for (i = 1; i < num_procs && chunks_sent < num_chunks; i++) {
        MPI_Send(&chunks[chunks_sent].chunk_id, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        MPI_Send(&chunks[chunks_sent].text_length, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        MPI_Send(chunks[chunks_sent].text, chunks[chunks_sent].text_length, MPI_CHAR, i, 0, MPI_COMM_WORLD);
        
        printf("Master: Sent chunk %d to process %d\n", chunks_sent, i);
        chunks_sent++;
    }
    
    while (chunks_received < num_chunks) {
        int chunk_id, summary_length;
        
        MPI_Recv(&chunk_id, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&summary_length, 1, MPI_INT, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(summaries[chunks_received].summary, summary_length, MPI_CHAR, status.MPI_SOURCE, 1, MPI_COMM_WORLD, &status);
        
        summaries[chunks_received].chunk_id = chunk_id;
        summaries[chunks_received].summary_length = summary_length;
        summaries[chunks_received].summary[summary_length] = '\0';
        
        printf("Master: Received summary for chunk %d from process %d\n", chunk_id, status.MPI_SOURCE);
        chunks_received++;
        
        if (chunks_sent < num_chunks) {
            MPI_Send(&chunks[chunks_sent].chunk_id, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            MPI_Send(&chunks[chunks_sent].text_length, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            MPI_Send(chunks[chunks_sent].text, chunks[chunks_sent].text_length, MPI_CHAR, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
            
            printf("Master: Sent chunk %d to process %d\n", chunks_sent, status.MPI_SOURCE);
            chunks_sent++;
        } else {
            int terminate_signal = -1;
            MPI_Send(&terminate_signal, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
        }
    }
    
    printf("\nAll chunks processed. Combining summaries...\n");
    
    char final_summary[MAX_SUMMARY_SIZE * 10];
    combine_summaries(summaries, num_chunks, final_summary, topic);
    
    end_time = MPI_Wtime();
    
    printf("\n=================================================\n");
    printf("FINAL SUMMARY\n");
    printf("=================================================\n");
    printf("%s\n", final_summary);
    printf("=================================================\n");
    printf("Total execution time: %.4f seconds\n", end_time - start_time);
    printf("Chunks processed: %d\n", num_chunks);
    printf("Average time per chunk: %.4f seconds\n", (end_time - start_time) / num_chunks);
    printf("=================================================\n");
    
    FILE *output = fopen("mpi_summary_output.txt", "w");
    if (output) {
        fprintf(output, "MPI Parallel Document Summarization Results\n");
        fprintf(output, "===========================================\n\n");
        fprintf(output, "Input File: %s\n", input_file);
        fprintf(output, "Topic: %s\n", topic);
        fprintf(output, "Number of Processes: %d\n", num_procs);
        fprintf(output, "Number of Chunks: %d\n", num_chunks);
        fprintf(output, "Execution Time: %.4f seconds\n\n", end_time - start_time);
        fprintf(output, "FINAL SUMMARY:\n");
        fprintf(output, "%s\n", final_summary);
        fclose(output);
        printf("\nResults saved to mpi_summary_output.txt\n");
    }
    
    free(chunks);
    free(summaries);
}

void worker_process(int rank, char *topic) {
    MPI_Status status;
    int chunk_id, text_length;
    char chunk_text[MAX_CHUNK_SIZE];
    char summary[MAX_SUMMARY_SIZE];
    
    while (1) {
        MPI_Recv(&chunk_id, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        
        if (chunk_id == -1) {
            printf("Process %d: Received termination signal\n", rank);
            break;
        }
        
        MPI_Recv(&text_length, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(chunk_text, text_length, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        chunk_text[text_length] = '\0';
        
        printf("Process %d: Processing chunk %d (%d bytes)\n", rank, chunk_id, text_length);
        
        call_python_summarizer(chunk_text, topic, summary);
        
        int summary_length = strlen(summary);
        
        MPI_Send(&chunk_id, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(&summary_length, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
        MPI_Send(summary, summary_length, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
        
        printf("Process %d: Sent summary for chunk %d (%d bytes)\n", rank, chunk_id, summary_length);
    }
}

int read_document_chunks(char *filename, DocumentChunk **chunks) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        printf("Error: Cannot open file %s\n", filename);
        return -1;
    }
    
    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);
    fseek(file, 0, SEEK_SET);
    
    char *full_text = (char *)malloc(file_size + 1);
    fread(full_text, 1, file_size, file);
    full_text[file_size] = '\0';
    fclose(file);
    
    int chunk_size = 2000;
    int num_chunks = (file_size + chunk_size - 1) / chunk_size;
    
    *chunks = (DocumentChunk *)malloc(num_chunks * sizeof(DocumentChunk));
    
    int chunk_idx = 0;
    for (int i = 0; i < file_size; i += chunk_size) {
        int current_chunk_size = (i + chunk_size > file_size) ? (file_size - i) : chunk_size;
        
        strncpy((*chunks)[chunk_idx].text, full_text + i, current_chunk_size);
        (*chunks)[chunk_idx].text[current_chunk_size] = '\0';
        (*chunks)[chunk_idx].chunk_id = chunk_idx;
        (*chunks)[chunk_idx].text_length = current_chunk_size;
        
        chunk_idx++;
    }
    
    free(full_text);
    return num_chunks;
}

void call_python_summarizer(char *chunk_text, char *topic, char *output_summary) {
    char temp_input_file[256];
    char temp_output_file[256];
    
    sprintf(temp_input_file, "temp_chunk_%d.txt", getpid());
    sprintf(temp_output_file, "temp_summary_%d.txt", getpid());
    
    FILE *input = fopen(temp_input_file, "w");
    if (input) {
        fprintf(input, "%s", chunk_text);
        fclose(input);
    }
    
    char command[1024];
    sprintf(command, "python mpi_python_wrapper.py \"%s\" \"%s\" \"%s\"", 
            temp_input_file, topic, temp_output_file);
    
    system(command);
    
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
    sprintf(combined_file, "temp_combined_%d.txt", getpid());
    
    FILE *combined = fopen(combined_file, "w");
    if (combined) {
        for (int i = 0; i < num_summaries; i++) {
            fprintf(combined, "%s\n\n", summaries[i].summary);
        }
        fclose(combined);
    }
    
    char output_file[256];
    sprintf(output_file, "temp_final_%d.txt", getpid());
    
    char command[1024];
    sprintf(command, "python mpi_final_combiner.py \"%s\" \"%s\" \"%s\"", 
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
