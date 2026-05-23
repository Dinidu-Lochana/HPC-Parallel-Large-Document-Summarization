/*
 * OpenMP Multithreaded Document Summarizer
 * Pattern : parallel-for with dynamic scheduling
 * Build   : gcc -O2 -Wall -fopenmp -o bin/openmp_summarizer openmp/openmp_summarizer.c
 * Run     : ./bin/openmp_summarizer <doc.txt> "<topic>" <num_threads>
 *           (must be run from project root)
 *
 * Note    : system() is thread-safe on POSIX/glibc; each call uses unique
 *           temp files keyed by (thread_id, chunk_id) to avoid conflicts.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <unistd.h>     /* sysconf */

/* ── tunables ────────────────────────────────────────────────────────────── */
#define CHUNK_SIZE   2000
#define MAX_CONTENT  4096
#define MAX_SUMMARY  2048
#define MAX_CHUNKS   500

/* ── globals ─────────────────────────────────────────────────────────────── */
static int  n_chunks;
static char chunk_data[MAX_CHUNKS][MAX_CONTENT];
static char summaries [MAX_CHUNKS][MAX_SUMMARY];   /* index i written by one thread */

/* ── helpers ─────────────────────────────────────────────────────────────── */

static int read_chunks(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) { perror(path); return -1; }
    int cnt = 0;
    while (!feof(f) && cnt < MAX_CHUNKS) {
        size_t n = fread(chunk_data[cnt], 1, CHUNK_SIZE, f);
        if (n == 0) break;
        chunk_data[cnt][n] = '\0';
        cnt++;
    }
    fclose(f);
    return cnt;
}

static void safe_topic(const char *t, char *out, size_t max)
{
    size_t j = 0;
    for (size_t i = 0; t[i] && j + 1 < max; i++)
        if (t[i] != '\'') out[j++] = t[i];
    out[j] = '\0';
}

/*
 * Thread-safe summarizer:
 *   - each (tid, cid) pair produces unique temp file names
 *   - no two OMP iterations share the same chunk index (schedule guarantees this)
 *   - returns chunk processing time via *elapsed
 */
static void call_wrapper(int tid, int cid, const char *text,
                         const char *topic, char *summary, double *elapsed)
{
    char in_f[256], out_f[256], cmd[1600], stopic[256];

    const char *td = getenv("TMPDIR");
    if (!td) td = getenv("TMP");
    if (!td) td = getenv("TEMP");
    if (!td) td = "/tmp";

    snprintf(in_f,  sizeof(in_f),  "%s/hpc_omp_in_%d_%d.txt",  td, tid, cid);
    snprintf(out_f, sizeof(out_f), "%s/hpc_omp_out_%d_%d.txt", td, tid, cid);

    FILE *f = fopen(in_f, "w");
    if (!f) {
        snprintf(summary, MAX_SUMMARY, "[ERR: cannot open %s]", in_f);
        *elapsed = 0.0;
        return;
    }
    fputs(text, f);
    fclose(f);

    safe_topic(topic, stopic, sizeof(stopic));
    snprintf(cmd, sizeof(cmd),
             "python3 ./shared/chunk_wrapper.py '%s' '%s' '%s'",
             in_f, stopic, out_f);

    double t0 = omp_get_wtime();
    int rc    = system(cmd);        /* POSIX: system() is thread-safe on glibc */
    *elapsed  = omp_get_wtime() - t0;

    if (rc != 0) {
        snprintf(summary, MAX_SUMMARY, "[wrapper exit %d for chunk %d]", rc, cid);
    } else {
        FILE *g = fopen(out_f, "r");
        if (!g) {
            snprintf(summary, MAX_SUMMARY, "[no output for chunk %d]", cid);
        } else {
            size_t n = fread(summary, 1, MAX_SUMMARY - 1, g);
            summary[n] = '\0';
            fclose(g);
        }
    }
    remove(in_f);
    remove(out_f);
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <doc.txt> \"<topic>\" <num_threads>\n", argv[0]);
        return 1;
    }

    const char *doc      = argv[1];
    const char *topic    = argv[2];
    int         nthreads = atoi(argv[3]);
    if (nthreads < 1) nthreads = 1;

    n_chunks = read_chunks(doc);
    if (n_chunks <= 0) {
        fprintf(stderr, "Error: could not read document '%s'.\n", doc);
        return 1;
    }

    omp_set_num_threads(nthreads);
    printf("[OpenMP] threads=%d | chunks=%d | topic=\"%s\"\n",
           nthreads, n_chunks, topic);

    double wall_start = omp_get_wtime();
    double seq_total  = 0.0;

    /*
     * Each iteration is independent:
     *   - writes to summaries[i] (unique index per iteration)
     *   - uses unique temp files keyed by (tid, i)
     *   - seq_total accumulated via OpenMP reduction
     */
    #pragma omp parallel for schedule(dynamic) reduction(+:seq_total)
    for (int i = 0; i < n_chunks; i++) {
        int    tid     = omp_get_thread_num();
        double elapsed = 0.0;
        call_wrapper(tid, i, chunk_data[i], topic, summaries[i], &elapsed);
        seq_total += elapsed;
    }

    double wall = omp_get_wtime() - wall_start;

    /* write ordered summaries */
    FILE *sf = fopen("omp_summaries.txt", "w");
    if (sf) {
        for (int i = 0; i < n_chunks; i++)
            fprintf(sf, "=== Chunk %d ===\n%s\n\n", i + 1, summaries[i]);
        fclose(sf);
    }

    /* final combination */
    char cmd[512], stopic[256];
    safe_topic(topic, stopic, sizeof(stopic));
    snprintf(cmd, sizeof(cmd),
             "python3 ./shared/final_combiner.py omp_summaries.txt '%s' omp_output.txt",
             stopic);
    system(cmd);

    /* ── metrics ─────────────────────────────────────────────────────────── */
    double speedup  = (wall > 0.0) ? seq_total / wall : 1.0;
    double eff      = speedup / nthreads;
    long   n_cpu    = sysconf(_SC_NPROCESSORS_ONLN);
    double cpu_util = (n_cpu > 0) ? (100.0 * nthreads / n_cpu) : 0.0;
    if (cpu_util > 100.0) cpu_util = 100.0;

    printf("\n");
    printf("==================================================\n");
    printf("         OpenMP Performance Metrics               \n");
    printf("==================================================\n");
    printf("  OMP Threads          : %d\n", nthreads);
    printf("  Chunks Processed     : %d\n", n_chunks);
    printf("  Execution Time       : %.3f s\n", wall);
    printf("  Sequential Estimate  : %.3f s  (sum of chunk times)\n", seq_total);
    printf("  Speedup              : %.2fx\n", speedup);
    printf("  Efficiency           : %.1f%%\n", eff * 100.0);
    printf("  Scalability          : %.1f%% of ideal linear speedup\n", eff * 100.0);
    printf("  CPU Cores Available  : %ld\n", n_cpu);
    printf("  Resource Utilization : %.1f%% CPU  (%d / %ld cores active)\n",
           cpu_util, nthreads, n_cpu);
    printf("  Output               : omp_output.txt\n");
    printf("==================================================\n");

    return 0;
}
