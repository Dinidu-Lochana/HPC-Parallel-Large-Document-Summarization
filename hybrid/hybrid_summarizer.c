/*
 * Hybrid MPI + OpenMP Document Summarizer
 * Pattern : MPI distributes chunk ranges across nodes/processes;
 *           OpenMP parallelises each process's range with threads.
 *
 * Build   : mpicc -O2 -Wall -fopenmp -o bin/hybrid_summarizer hybrid/hybrid_summarizer.c
 * Run     : mpirun -np P ./bin/hybrid_summarizer <doc.txt> "<topic>" <threads_per_proc>
 *           (must be run from project root)
 *           Total parallelism = P × threads_per_proc
 *
 * Design  :
 *   1. All MPI processes read the document independently (no broadcast needed).
 *   2. Each process is assigned a static slice of chunks: [my_start, my_end).
 *   3. OpenMP parallel-for processes that slice concurrently with nthreads threads.
 *   4. MPI_Barrier syncs all processes.
 *   5. Workers send their summaries to rank 0 via MPI point-to-point.
 *   6. Rank 0 assembles the ordered summary file and calls the Python combiner.
 *   7. Timing via MPI_Reduce (max wall-time, sum seq-time) for accurate metrics.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>     /* sysconf */

/* ── tunables ────────────────────────────────────────────────────────────── */
#define CHUNK_SIZE   2000
#define MAX_CONTENT  4096
#define MAX_SUMMARY  2048
#define MAX_CHUNKS   500

/* ── globals (every process uses these) ─────────────────────────────────── */
static int  n_chunks;
static char chunk_data[MAX_CHUNKS][MAX_CONTENT];
static char summaries [MAX_CHUNKS][MAX_SUMMARY];   /* index i is unique per OMP iter */

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
 * Thread-safe summarizer — unique temp files keyed by (rank, tid, cid).
 * MPI rank ensures no two MPI processes collide; tid within the same process;
 * cid across iterations of the same thread (impossible with static/dynamic but
 * included for safety).
 */
static void call_wrapper(int rank, int tid, int cid,
                         const char *text, const char *topic,
                         char *summary, double *elapsed)
{
    char in_f[256], out_f[256], cmd[1600], stopic[256];

    const char *td = getenv("TMPDIR");
    if (!td) td = getenv("TMP");
    if (!td) td = getenv("TEMP");
    if (!td) td = "/tmp";

    snprintf(in_f,  sizeof(in_f),  "%s/hpc_hyb_in_%d_%d_%d.txt",  td, rank, tid, cid);
    snprintf(out_f, sizeof(out_f), "%s/hpc_hyb_out_%d_%d_%d.txt", td, rank, tid, cid);

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
    int rc    = system(cmd);
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
    /* MPI_THREAD_FUNNELED: only the master thread calls MPI after Init */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 4) {
        if (!rank)
            fprintf(stderr,
                    "Usage: mpirun -np P %s <doc.txt> \"<topic>\" <threads_per_proc>\n",
                    argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *doc      = argv[1];
    const char *topic    = argv[2];
    int         nthreads = atoi(argv[3]);
    if (nthreads < 1) nthreads = 1;

    /* ── step 1: all processes read the document independently ───────────── */
    n_chunks = read_chunks(doc);
    if (n_chunks <= 0) {
        fprintf(stderr, "Process %d: could not read document '%s'.\n", rank, doc);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* ── step 2: static distribution — process rank gets [my_start, my_end) */
    int base     = n_chunks / nprocs;
    int extra    = n_chunks % nprocs;               /* first `extra` procs get +1 */
    int my_start = rank * base + (rank < extra ? rank : extra);
    int my_end   = my_start + base + (rank < extra ? 1 : 0);

    omp_set_num_threads(nthreads);

    if (!rank)
        printf("[Hybrid] procs=%d | threads/proc=%d | total_parallelism=%d | "
               "chunks=%d | topic=\"%s\"\n",
               nprocs, nthreads, nprocs * nthreads, n_chunks, topic);

    /* ── step 3: synchronised start — all processes begin together ────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    double wall_start = omp_get_wtime();
    double local_seq  = 0.0;

    /*
     * OpenMP parallel-for over this process's slice.
     * Each (rank, tid, i) triple is globally unique → no temp-file conflict.
     */
    #pragma omp parallel for schedule(dynamic) reduction(+:local_seq)
    for (int i = my_start; i < my_end; i++) {
        int    tid     = omp_get_thread_num();
        double elapsed = 0.0;
        call_wrapper(rank, tid, i, chunk_data[i], topic, summaries[i], &elapsed);
        local_seq += elapsed;
    }

    double local_wall = omp_get_wtime() - wall_start;

    /* ── step 4: reduce timing across all processes ───────────────────────── */
    MPI_Barrier(MPI_COMM_WORLD);
    double max_wall  = 0.0, total_seq = 0.0;
    MPI_Reduce(&local_wall, &max_wall,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_seq,  &total_seq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* ── step 5: collect summaries at rank 0 ─────────────────────────────── */
    /*
     * Workers send their per-chunk summaries as one concatenated string.
     * Master receives in rank order, which equals chunk order because of the
     * static distribution above.
     */
    if (rank != 0) {
        /* build this process's summary block */
        int   buf_sz = (my_end - my_start) * (MAX_SUMMARY + 32) + 64;
        char *buf    = (char *)malloc(buf_sz);
        if (!buf) { MPI_Finalize(); return 1; }
        int   len = 0;
        for (int i = my_start; i < my_end; i++) {
            int written = snprintf(buf + len, buf_sz - len,
                                   "=== Chunk %d ===\n%s\n\n", i + 1, summaries[i]);
            if (written > 0) len += written;
        }
        MPI_Send(&len, 1, MPI_INT, 0, 20, MPI_COMM_WORLD);
        MPI_Send(buf,  len, MPI_CHAR, 0, 21, MPI_COMM_WORLD);
        free(buf);
    } else {
        /* rank 0: write own slice first (chunk 0 … my_end-1) */
        FILE *sf = fopen("hybrid_summaries.txt", "w");
        if (!sf) { MPI_Finalize(); return 1; }

        for (int i = my_start; i < my_end; i++)
            fprintf(sf, "=== Chunk %d ===\n%s\n\n", i + 1, summaries[i]);

        /* receive from ranks 1, 2, … in order (preserves chunk order) */
        for (int r = 1; r < nprocs; r++) {
            int recv_len = 0;
            MPI_Recv(&recv_len, 1, MPI_INT, r, 20, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            char *rbuf = (char *)malloc(recv_len + 1);
            if (!rbuf) continue;
            MPI_Recv(rbuf, recv_len, MPI_CHAR, r, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            rbuf[recv_len] = '\0';
            fputs(rbuf, sf);
            free(rbuf);
        }
        fclose(sf);

        /* ── step 6: final combination via Python ──────────────────────── */
        char cmd[512], stopic[256];
        safe_topic(topic, stopic, sizeof(stopic));
        snprintf(cmd, sizeof(cmd),
                 "python3 ./shared/final_combiner.py hybrid_summaries.txt '%s' hybrid_output.txt",
                 stopic);
        system(cmd);

        /* ── metrics ───────────────────────────────────────────────────── */
        int    total_units = nprocs * nthreads;
        double speedup     = (max_wall > 0.0) ? total_seq / max_wall : 1.0;
        double eff         = (total_units > 0) ? speedup / total_units : 1.0;
        long   n_cpu       = sysconf(_SC_NPROCESSORS_ONLN);
        double cpu_util    = (n_cpu > 0) ? (100.0 * total_units / n_cpu) : 0.0;
        if (cpu_util > 100.0) cpu_util = 100.0;

        printf("\n");
        printf("==================================================\n");
        printf("      Hybrid MPI+OpenMP Performance Metrics       \n");
        printf("==================================================\n");
        printf("  MPI Processes        : %d\n", nprocs);
        printf("  OMP Threads/Process  : %d\n", nthreads);
        printf("  Total Parallelism    : %d  (procs x threads)\n", total_units);
        printf("  Chunks Processed     : %d\n", n_chunks);
        printf("  Execution Time       : %.3f s  (max across all procs)\n", max_wall);
        printf("  Sequential Estimate  : %.3f s  (sum of chunk times)\n", total_seq);
        printf("  Speedup              : %.2fx\n", speedup);
        printf("  Efficiency           : %.1f%%\n", eff * 100.0);
        printf("  Scalability          : %.1f%% of ideal linear speedup\n", eff * 100.0);
        printf("  CPU Cores Available  : %ld\n", n_cpu);
        printf("  Resource Utilization : %.1f%% CPU  (%d / %ld cores active)\n",
               cpu_util, total_units, n_cpu);
        printf("  Output               : hybrid_output.txt\n");
        printf("==================================================\n");
    }

    MPI_Finalize();
    return 0;
}
