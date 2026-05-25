/*
 * MPI Parallel Document Summarizer
 * Pattern : Dynamic master-worker
 * Build   : mpicc -O2 -Wall -o bin/mpi_summarizer mpi/mpi_summarizer.c
 * Run     : mpirun -np 4 ./bin/mpi_summarizer <doc.txt> "<topic>"
 *           (must be run from project root)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h>     /* sysconf */

/* ── tunables ────────────────────────────────────────────────────────────── */
#define CHUNK_SIZE   2000   /* bytes per document chunk                       */
#define MAX_CONTENT  4096   /* chunk buffer (slightly larger than CHUNK_SIZE) */
#define MAX_SUMMARY  2048   /* summary buffer                                 */
#define MAX_CHUNKS   500    /* maximum number of chunks                       */
#define TAG_WORK     1
#define TAG_RESULT   2

/* ── fixed-size message structs (sent as MPI_BYTE) ───────────────────────── */
typedef struct { int id; char text[MAX_CONTENT]; } WorkMsg;   /* id=-1 → quit */
typedef struct { int id; double elapsed; char text[MAX_SUMMARY]; } ResultMsg;

/* ── globals (master only needs chunk_data; results stored here) ─────────── */
static int       n_chunks;
static char      chunk_data[MAX_CHUNKS][MAX_CONTENT];
static ResultMsg g_results[MAX_CHUNKS];

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

/* strip single-quotes so the topic can be safely embedded in a shell command */
static void safe_topic(const char *t, char *out, size_t max)
{
    size_t j = 0;
    for (size_t i = 0; t[i] && j + 1 < max; i++)
        if (t[i] != '\'') out[j++] = t[i];
    out[j] = '\0';
}

/* call chunk_wrapper.py; uses rank+cid for unique temp-file names */
static void call_wrapper(int rank, int cid, const char *text,
                         const char *topic, char *summary, double *elapsed)
{
    char in_f[256], out_f[256], cmd[1600], stopic[256];

    /* temp directory: honour $TMPDIR, fall back to /tmp */
    const char *td = getenv("TMPDIR");
    if (!td) td = getenv("TMP");
    if (!td) td = getenv("TEMP");
    if (!td) td = "/tmp";

    snprintf(in_f,  sizeof(in_f),  "%s/hpc_mpi_in_%d_%d.txt",  td, rank, cid);
    snprintf(out_f, sizeof(out_f), "%s/hpc_mpi_out_%d_%d.txt", td, rank, cid);

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

    double t0 = MPI_Wtime();
    int rc    = system(cmd);
    *elapsed  = MPI_Wtime() - t0;

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

/* ── master process ──────────────────────────────────────────────────────── */

static void run_master(int nprocs, const char *topic)
{
    double wall_start = MPI_Wtime();
    double seq_total  = 0.0;

    int      next = 0, active = 0;
    WorkMsg  wm;
    ResultMsg rm;
    MPI_Status st;

    /* seed each worker with its first chunk */
    for (int w = 1; w < nprocs && next < n_chunks; w++, next++, active++) {
        wm.id = next;
        strncpy(wm.text, chunk_data[next], MAX_CONTENT - 1);
        wm.text[MAX_CONTENT - 1] = '\0';
        MPI_Send(&wm, sizeof(WorkMsg), MPI_BYTE, w, TAG_WORK, MPI_COMM_WORLD);
    }

    /* dynamic scheduling: send next chunk as soon as a result arrives */
    while (active > 0) {
        MPI_Recv(&rm, sizeof(ResultMsg), MPI_BYTE,
                 MPI_ANY_SOURCE, TAG_RESULT, MPI_COMM_WORLD, &st);
        active--;
        g_results[rm.id] = rm;
        seq_total += rm.elapsed;

        if (next < n_chunks) {
            wm.id = next;
            strncpy(wm.text, chunk_data[next], MAX_CONTENT - 1);
            wm.text[MAX_CONTENT - 1] = '\0';
            MPI_Send(&wm, sizeof(WorkMsg), MPI_BYTE,
                     st.MPI_SOURCE, TAG_WORK, MPI_COMM_WORLD);
            next++;
            active++;
        }
    }

    /* terminate all workers */
    wm.id = -1;
    wm.text[0] = '\0';
    for (int w = 1; w < nprocs; w++)
        MPI_Send(&wm, sizeof(WorkMsg), MPI_BYTE, w, TAG_WORK, MPI_COMM_WORLD);

    /* write ordered summaries file */
    FILE *sf = fopen("mpi_summaries.txt", "w");
    if (sf) {
        for (int i = 0; i < n_chunks; i++)
            fprintf(sf, "=== Chunk %d ===\n%s\n\n", i + 1, g_results[i].text);
        fclose(sf);
    }

    /* final combination via Python */
    char cmd[512], stopic[256];
    safe_topic(topic, stopic, sizeof(stopic));
    snprintf(cmd, sizeof(cmd),
             "python3 ./shared/final_combiner.py mpi_summaries.txt '%s' mpi_output.txt",
             stopic);
    system(cmd);

    /* ── metrics ─────────────────────────────────────────────────────────── */
    double wall     = MPI_Wtime() - wall_start;
    int    workers  = nprocs - 1;
    double speedup  = (wall > 0.0 && workers > 0) ? seq_total / wall : 1.0;
    double eff      = (workers > 0) ? speedup / workers : 1.0;
    long   n_cpu    = sysconf(_SC_NPROCESSORS_ONLN);
    double cpu_util = (n_cpu > 0) ? (100.0 * workers / n_cpu) : 0.0;
    if (cpu_util > 100.0) cpu_util = 100.0;

    printf("\n");
    printf("==================================================\n");
    printf("           MPI Performance Metrics                \n");
    printf("==================================================\n");
    printf("  MPI Processes        : %d  (1 master + %d workers)\n", nprocs, workers);
    printf("  Chunks Processed     : %d\n", n_chunks);
    printf("  Execution Time       : %.3f s\n", wall);
    printf("  Sequential Estimate  : %.3f s  (sum of chunk times)\n", seq_total);
    printf("  Speedup              : %.2fx\n", speedup);
    printf("  Efficiency           : %.1f%%\n", eff * 100.0);
    printf("  Scalability          : %.1f%% of ideal linear speedup\n", eff * 100.0);
    printf("  CPU Cores Available  : %ld\n", n_cpu);
    printf("  Resource Utilization : %.1f%% CPU  (%d / %ld cores active)\n",
           cpu_util, workers, n_cpu);
    printf("  Output               : mpi_output.txt\n");
    printf("==================================================\n");
}

/* ── worker process ──────────────────────────────────────────────────────── */

static void run_worker(int rank, const char *topic)
{
    WorkMsg   wm;
    ResultMsg rm;
    MPI_Status st;

    while (1) {
        MPI_Recv(&wm, sizeof(WorkMsg), MPI_BYTE, 0, TAG_WORK, MPI_COMM_WORLD, &st);
        if (wm.id < 0) break;          /* termination signal */

        rm.id = wm.id;
        call_wrapper(rank, wm.id, wm.text, topic, rm.text, &rm.elapsed);

        MPI_Send(&rm, sizeof(ResultMsg), MPI_BYTE, 0, TAG_RESULT, MPI_COMM_WORLD);
    }
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    int rank, nprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 3) {
        if (!rank)
            fprintf(stderr, "Usage: mpirun -np N %s <doc.txt> \"<topic>\"\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    if (nprocs < 2) {
        if (!rank)
            fprintf(stderr, "Error: need at least 2 MPI processes (1 master + 1 worker).\n");
        MPI_Finalize();
        return 1;
    }

    const char *doc   = argv[1];
    const char *topic = argv[2];

    if (!rank) {
        n_chunks = read_chunks(doc);
        if (n_chunks <= 0) {
            fprintf(stderr, "Error: could not read document '%s'.\n", doc);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("[MPI] processes=%d | chunks=%d | topic=\"%s\"\n",
               nprocs, n_chunks, topic);
    }

    /* workers don't need n_chunks — they receive work until id == -1 */
    if (!rank) run_master(nprocs, topic);
    else        run_worker(rank, topic);

    MPI_Finalize();
    return 0;
}
