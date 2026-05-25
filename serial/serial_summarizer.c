/*
 * Serial Document Summarizer — baseline for speedup comparison
 * Build : gcc -O2 -Wall -o bin/serial_summarizer serial/serial_summarizer.c
 * Run   : ./bin/serial_summarizer <doc.txt> "<topic>"
 *         (must be run from project root)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>     /* sysconf */

/* ── tunables ────────────────────────────────────────────────────────────── */
#define CHUNK_SIZE   2000
#define MAX_CONTENT  4096
#define MAX_SUMMARY  2048
#define MAX_CHUNKS   500

/* ── globals ─────────────────────────────────────────────────────────────── */
static int  n_chunks;
static char chunk_data[MAX_CHUNKS][MAX_CONTENT];
static char summaries [MAX_CHUNKS][MAX_SUMMARY];

/* ── helpers ─────────────────────────────────────────────────────────────── */

static double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

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

static void call_wrapper(int cid, const char *text,
                         const char *topic, char *summary, double *elapsed)
{
    char in_f[256], out_f[256], cmd[1600], stopic[256];

    const char *td = getenv("TMPDIR");
    if (!td) td = getenv("TMP");
    if (!td) td = getenv("TEMP");
    if (!td) td = "/tmp";

    snprintf(in_f,  sizeof(in_f),  "%s/hpc_serial_in_%d.txt",  td, cid);
    snprintf(out_f, sizeof(out_f), "%s/hpc_serial_out_%d.txt", td, cid);

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

    double t0 = now_sec();
    int rc    = system(cmd);
    *elapsed  = now_sec() - t0;

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
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <doc.txt> \"<topic>\"\n", argv[0]);
        return 1;
    }

    const char *doc   = argv[1];
    const char *topic = argv[2];

    n_chunks = read_chunks(doc);
    if (n_chunks <= 0) {
        fprintf(stderr, "Error: could not read document '%s'.\n", doc);
        return 1;
    }

    printf("[Serial] chunks=%d | topic=\"%s\"\n", n_chunks, topic);

    double wall_start = now_sec();
    double seq_total  = 0.0;

    for (int i = 0; i < n_chunks; i++) {
        double elapsed = 0.0;
        call_wrapper(i, chunk_data[i], topic, summaries[i], &elapsed);
        seq_total += elapsed;
        printf("[Serial] Chunk %d/%d done in %.2f s\n", i + 1, n_chunks, elapsed);
    }

    double wall = now_sec() - wall_start;

    /* write ordered summaries */
    FILE *sf = fopen("serial_summaries.txt", "w");
    if (sf) {
        for (int i = 0; i < n_chunks; i++)
            fprintf(sf, "=== Chunk %d ===\n%s\n\n", i + 1, summaries[i]);
        fclose(sf);
    }

    /* final combination */
    char cmd[512], stopic[256];
    safe_topic(topic, stopic, sizeof(stopic));
    snprintf(cmd, sizeof(cmd),
             "python3 ./shared/final_combiner.py serial_summaries.txt '%s' serial_output.txt",
             stopic);
    int _rc = system(cmd); (void)_rc;

    /* ── metrics ─────────────────────────────────────────────────────────── */
    long   n_cpu    = sysconf(_SC_NPROCESSORS_ONLN);
    double cpu_util = (n_cpu > 0) ? (100.0 / n_cpu) : 100.0;
    if (cpu_util > 100.0) cpu_util = 100.0;

    printf("\n");
    printf("==================================================\n");
    printf("         Serial Performance Metrics               \n");
    printf("==================================================\n");
    printf("  Chunks Processed     : %d\n",    n_chunks);
    printf("  Execution Time       : %.3f s\n", wall);
    printf("  Sequential Estimate  : %.3f s  (sum of chunk times)\n", seq_total);
    printf("  Speedup              : 1.00x\n");
    printf("  Efficiency           : 100.0%%\n");
    printf("  Scalability          : 100.0%%\n");
    printf("  CPU Cores Available  : %ld\n",   n_cpu);
    printf("  Resource Utilization : %.1f%% CPU  (1 / %ld cores active)\n",
           cpu_util, n_cpu);
    printf("  Output               : serial_output.txt\n");
    printf("==================================================\n");

    return 0;
}
