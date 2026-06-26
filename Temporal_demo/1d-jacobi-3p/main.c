#include "defines_512.h"

static double elapsed_seconds(struct timeval start, struct timeval end)
{
    return (double)(end.tv_sec - start.tv_sec) +
           (double)(end.tv_usec - start.tv_usec) * 1.0e-6;
}

static void reset_input(int NX, double (*A)[NX + 2 * XSTART],
                        const double (*backup)[NX + 2 * XSTART])
{
    for (int x = 0; x < NX + 2 * XSTART; x++) {
        A[0][x] = backup[0][x];
        A[1][x] = backup[0][x];
    }
}

int checkresult(int NX, double *A_correct, double *A)
{
    int correct = 1;

    for (int x = XSTART; x < NX + XSTART; x++) {
        if (A_correct[x] != A[x]) {
            if (correct) {
                printf("First mismatch: ");
            }
            printf("x = [%d], Correct = %.17g, Wrong = %.17g\n",
                   x, A_correct[x], A[x]);
            correct = 0;
            break;
        }
    }

    return correct;
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s NX T\n", argv[0]);
        return 1;
    }

    int NX = atoi(argv[1]);
    int T = atoi(argv[2]);
    if (NX <= STRIDE * VVECLEN || T <= 0) {
        fprintf(stderr, "NX must be > %d and T must be > 0.\n", STRIDE * VVECLEN);
        return 1;
    }

    double (*A)[NX + 2 * XSTART] =
        (double (*)[NX + 2 * XSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * 2);
    double (*A_backup)[NX + 2 * XSTART] =
        (double (*)[NX + 2 * XSTART])malloc(sizeof(double) * (NX + 2 * XSTART));
    double (*A_correct)[NX + 2 * XSTART] =
        (double (*)[NX + 2 * XSTART])malloc(sizeof(double) * (NX + 2 * XSTART) * 2);

    if (!A || !A_backup || !A_correct) {
        fprintf(stderr, "Allocation failed.\n");
        free(A);
        free(A_backup);
        free(A_correct);
        return 1;
    }

    srand(100);
    for (int x = 0; x < NX + 2 * XSTART; x++) {
        A_backup[0][x] = INIT;
    }

    struct timeval start, end;

    reset_input(NX, A_correct, A_backup);
    gettimeofday(&start, 0);
    naive_scalar((double *)A_correct, NX, T);
    gettimeofday(&end, 0);
    double ref_time = elapsed_seconds(start, end);

    reset_input(NX, A, A_backup);
    gettimeofday(&start, 0);
    vectime_stride7_512((double *)A, NX, T);
    gettimeofday(&end, 0);
    double opt_time = elapsed_seconds(start, end);

    int ok = checkresult(NX, &A_correct[T % 2][0], &A[T % 2][0]);
    printf("%s\tvectime_stride7_512, NX = %d, T = %d, GStencil/s = %f\n",
           ok ? "Correct!" : "Wrong!", NX, T, ((double)NX * T) / opt_time / 1e9);
    printf("reference naive_scalar, time = %.6f s, GStencil/s = %f\n",
           ref_time, ((double)NX * T) / ref_time / 1e9);

    free(A_correct);
    free(A);
    free(A_backup);

    return ok ? 0 : 2;
}
