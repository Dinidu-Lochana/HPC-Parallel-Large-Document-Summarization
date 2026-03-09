// omp_summarizer.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/stat.h>
#include <errno.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#define POPEN _popen
#define PCLOSE _pclose
#define PYTHON_CMD "python"
#else
#define MKDIR(path) mkdir(path, 0755)
#define POPEN popen
#define PCLOSE pclose
#define PYTHON_CMD "python3"
#endif

#define MAX_PATH   512
#define MAX_CMD    1024

static int ensure_dir(const char *path) {
    if (MKDIR(path) == 0 || errno == EEXIST) {
        return 0;
    }
    perror(path);
    return -1;
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: %s <pdf_path> <file_name> <topic> <num_threads>\n", argv[0]);
        printf("Example: %s doc.pdf \"doc.pdf\" \"Machine Learning\" 4\n", argv[0]);
        return 1;
    }

    const char *pdf_path    = argv[1];
    const char *file_name   = argv[2];
    const char *topic       = argv[3];
    int         num_threads = atoi(argv[4]);

    const char *chunks_dir    = "chunks/openmp_chunks";
    const char *summaries_dir = "summaries/openmp";

    if (ensure_dir("chunks") != 0) return 1;
    if (ensure_dir("summaries") != 0) return 1;
    if (ensure_dir(chunks_dir) != 0) return 1;
    if (ensure_dir(summaries_dir) != 0) return 1;

    printf("[Main] Splitting PDF into chunks...\n");
    char split_cmd[MAX_CMD];
    snprintf(split_cmd, MAX_CMD,
             "%s python_llm/split_pdf.py \"%s\" 2000 \"%s\"",
             PYTHON_CMD,
             pdf_path, chunks_dir);

    FILE *pipe = POPEN(split_cmd, "r");
    if (!pipe) {
        perror("popen split");
        return 1;
    }

    int num_chunks = 0;
    fscanf(pipe, "%d", &num_chunks);
    PCLOSE(pipe);

    if (num_chunks <= 0) {
        fprintf(stderr, "[Error] No chunks produced.\n");
        return 1;
    }

    printf("[Main] %d chunks created. Running with %d threads.\n\n", num_chunks, num_threads);

    double t_start = omp_get_wtime();
    int i;

    #pragma omp parallel for schedule(dynamic) num_threads(num_threads)
    for (i = 0; i < num_chunks; i++) {
        int tid = omp_get_thread_num();

        char chunk_path[MAX_PATH];
        char summary_path[MAX_PATH];
        char cmd[MAX_CMD];

        snprintf(chunk_path, MAX_PATH, "%s/chunk_%d.txt", chunks_dir, i);
        snprintf(summary_path, MAX_PATH, "%s/summary_%d.txt", summaries_dir, i);

        snprintf(cmd, MAX_CMD,
                 "%s python_llm/summarize_chunk_cli.py \"%s\" \"%s\" \"%s\" \"%s\"",
                 PYTHON_CMD,
                 chunk_path, file_name, topic, summary_path);

        printf("[Thread %d] Starting chunk %d/%d\n", tid, i + 1, num_chunks);
        double t0 = omp_get_wtime();

        int ret = system(cmd);

        double t1 = omp_get_wtime();
        if (ret == 0)
            printf("[Thread %d] Finished chunk %d/%d in %.1fs\n", tid, i + 1, num_chunks, t1 - t0);
        else
            printf("[Thread %d] ERROR on chunk %d\n", tid, i + 1);
    }

    double t_parallel_end = omp_get_wtime();

    printf("\n[Main] All chunks done. Generating final summary...\n");
    char final_cmd[MAX_CMD];
    snprintf(final_cmd, MAX_CMD,
             "%s python_llm/final_summary_cli.py \"%s\" \"%s\" \"%s\" \"summaries/openmp/final_summary.txt\"",
             PYTHON_CMD,
             summaries_dir, file_name, topic);
    system(final_cmd);

    double t_end = omp_get_wtime();

    printf("\n========== PERFORMANCE METRICS ==========\n");
    printf("Threads used:       %d\n", num_threads);
    printf("Chunks processed:   %d\n", num_chunks);
    printf("Parallel time:      %.2fs\n", t_parallel_end - t_start);
    printf("Final summary time: %.2fs\n", t_end - t_parallel_end);
    printf("Total time:         %.2fs\n", t_end - t_start);
    printf("Output:             summaries/openmp/final_summary.txt\n");
    printf("=========================================\n");

    return 0;
}
