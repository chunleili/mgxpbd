#define MAX_MSG_LEN 4096

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* CUDA error macro */
#define CUDA_SAFE_CALL(call) do {                                 \
  cudaError_t err = call;                                         \
  if(cudaSuccess != err) {                                        \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
            __FILE__, __LINE__, cudaGetErrorString( err) );       \
    exit(EXIT_FAILURE);                                           \
  } } while (0)

/* standard or dynamically load library */
#ifdef AMGX_DYNAMIC_LOADING
#include "amgx_capi.h"
#else
#include "amgx_c.h"
#endif

/* print error message and exit */
void errAndExit(const char *err)
{
    printf("%s\n", err);
    fflush(stdout);
    exit(1);
}

/* print callback (could be customized) */
void print_callback(const char *msg, int length)
{
    printf("%s", msg);
}

/* print usage and exit */
void printUsageAndExit()
{
    printf("%s", "Usage: ./amgx_capi [-mode [hDDI | hDFI | hFFI | dDDI | dDFI | dFFI]] [-m file] [-c config_file] [-amg \"variable1=value1 ... variable3=value3\"]\n");
    printf("%s", "     -mode:   select the solver mode\n");
    printf("%s", "     -m file: read matrix stored in the file\n");
    printf("%s", "     -c:      set the amg solver options from the config file\n");
    printf("%s", "     -amg:    set the amg solver options from the command line\n");
    exit(0);
}

/* parse parameters */
int findParamIndex(const char **argv, int argc, const char *parm)
{
    int count = 0;
    int index = -1;

    for (int i = 0; i < argc; i++)
    {
        if (strncmp(argv[i], parm, 100) == 0)
        {
            index = i;
            count++;
        }
    }

    if (count == 0 || count == 1)
    {
        return index;
    }
    else
    {
        printf("Error, parameter %s has been specified more than once, exiting\n", parm);
        exit(1);
    }

    return -1;
}

/* reade geometry (advanced input parmeters) */
void readGeometry( const char *fname, double **geo_x, double **geo_y, double **geo_z, int *dim, int *numrows)
{
    printf("Reading geometry from file: '%s'\n", fname);
    FILE *fin = fopen(fname, "r");

    if (!fin)
    {
        printf("Error opening file '%s'\n", fname);
        exit(1);
    }

    int n, dimension;

    if (2 != fscanf(fin, "%d %d\n", &n, &dimension))
    {
        errAndExit("Bad format\n");
    }

    *geo_x = (double *)malloc(n * sizeof(double));
    *geo_y = (double *)malloc(n * sizeof(double));

    if (dimension == 3)
    {
        *geo_y = (double *)malloc(n * sizeof(double));

        for (int i = 0; i < n; i ++)
            if (3 != fscanf(fin, "%lf %lf %lf\n", *geo_x + i, *geo_y + i, *geo_z + i))
            {
                errAndExit("Bad format\n");
            }
    }
    else if (dimension == 2)
    {
        for (int i = 0; i < n; i ++)
            if ( 2 != fscanf(fin, "%lf %lf\n", *geo_x + i, *geo_y + i))
            {
                errAndExit("Bad format\n");
            }
    }

    *dim = dimension;
    *numrows = n;
}

/* reade coloring (advanced input parmeters) */
void readColoring( const char *fname, int **row_coloring, int *colored_rows, int *num_colors)
{
    printf("Reading coloring from file: '%s'\n", fname);
    FILE *fin = fopen(fname, "r");

    if (!fin)
    {
        printf("Error opening file '%s'\n", fname);
        exit(1);
    }

    int n, colors_num;

    if (2 != fscanf(fin, "%d %d\n", &n, &colors_num))
    {
        errAndExit("Bad format\n");
    }

    *row_coloring = (int *)malloc(n * sizeof(int));

    for (int i = 0; i < n; i ++)
        if ( 1 != fscanf(fin, "%d\n", *row_coloring + i))
        {
            errAndExit("Bad format\n");
        }

    *colored_rows = n;
    *num_colors = colors_num;
}


int main(int argc, const char **argv)
{
    //parameter parsing
    int pidx = 0;
    int pidy = 0;
    //versions
    int major, minor;
    char *ver, *date, *time;
    //input geometry
    double *gx = NULL;
    double *gy = NULL;
    double *gz = NULL;
    //input coloring
    int dim = 0;
    int numrows = 0;
    int num_colors = 0;
    int colored_rows = 0;
    int *row_coloring = NULL;
    //input matrix and rhs/solution
    int n = 0;
    int bsize_x = 0;
    int bsize_y = 0;
    int sol_size = 0;
    int sol_bsize = 0;
    //library handles
    AMGX_Mode mode;
    AMGX_config_handle cfg;
    AMGX_resources_handle rsrc;
    AMGX_matrix_handle A;
    AMGX_vector_handle b, x;
    AMGX_solver_handle solver;
    //status handling
    AMGX_SOLVE_STATUS status;

    /* check arguments */
    if (argc == 1)
    {
        printUsageAndExit();
    }

    /* init */
    AMGX_SAFE_CALL(AMGX_initialize());
    /* system */
    AMGX_SAFE_CALL(AMGX_register_print_callback(&print_callback));
    AMGX_SAFE_CALL(AMGX_install_signal_handler());

    mode = AMGX_mode_dDDI; //8193: double vec, double mat, int ind, device

    /* create config */
    pidx = findParamIndex(argv, argc, "-amg");
    pidy = findParamIndex(argv, argc, "-c");

    AMGX_SAFE_CALL(AMGX_config_create_from_file(&cfg, argv[pidy + 1]));

    AMGX_resources_create_simple(&rsrc, cfg);
    AMGX_matrix_create(&A, rsrc, mode);
    AMGX_vector_create(&x, rsrc, mode);
    AMGX_vector_create(&b, rsrc, mode);
    AMGX_solver_create(&solver, rsrc, mode, cfg);

    /* read the input system: matrix [and rhs & solution]
       Please refer to AMGX_read_system description in the AMGX_Reference.pdf
       manual for details on how to specify the rhs and the solution inside
       the input file. If these are not specified than rhs=[1,...,1]^T and
       (initial guess) sol=[0,...,0]^T. */
    if ((pidx = findParamIndex(argv, argc, "-m")) != -1)
    {
        AMGX_read_system(A, b, x, argv[pidx + 1]);
        AMGX_matrix_get_size(A, &n, &bsize_x, &bsize_y);
        AMGX_vector_get_size(x, &sol_size, &sol_bsize);

        if (sol_size == 0 || sol_bsize == 0)
        {
            AMGX_vector_set_zero(x, n, bsize_x);
        }
    }
    else
    {
        errAndExit("ERROR: no linear system was specified");
    }

// example of getting initial residual norm
    
      {
        void* t_norm = calloc(bsize_x, sizeof(double));
        AMGX_solver_calculate_residual_norm(solver, A, b, x, t_norm);
        printf("Initial norm: ");
        for (int i = 0; i < bsize_x; i++)
        {
          printf("%f ", ((double*)t_norm)[i]);
        }
        printf("\n");
        free(t_norm);
      }
    

    /* solver setup */
    AMGX_solver_setup(solver, A);
    /* solver solve */
    AMGX_solver_solve(solver, b, x);
    /* example of how to change parameters between non-linear iterations */
    //AMGX_config_add_parameters(&cfg, "config_version=2, default:tolerance=1e-12");
    //AMGX_solver_solve(solver, b, x);
    AMGX_solver_get_status(solver, &status);
    /* example of how to print the residual history */
    int nit;
    double res;
    AMGX_solver_get_iterations_number(solver, &nit);
    for (int i=0; i<nit; i++) {
     printf("residual from iteration %d=", i);
     for (int j=0; j<bsize_y; j++) {
       AMGX_solver_get_iteration_residual(solver, i, j, &res);
       printf("%f ", (float)(res));
     }
     printf("\n");
    }
    /* example of how to write the linear system to the output */
    AMGX_write_system(A, b, x, "output.system.mtx");
    /* destroy resources, matrix, vector and solver */
    AMGX_solver_destroy(solver);
    AMGX_vector_destroy(x);
    AMGX_vector_destroy(b);
    AMGX_matrix_destroy(A);
    AMGX_resources_destroy(rsrc);
    /* destroy config (need to use AMGX_SAFE_CALL after this point) */
    AMGX_SAFE_CALL(AMGX_config_destroy(cfg));
    /* shutdown and exit */
    AMGX_SAFE_CALL(AMGX_finalize());
    /* close the library (if it was dynamically loaded) */
    return status;
}
