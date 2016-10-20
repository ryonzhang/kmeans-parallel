#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <string.h>


#define MAX_ITR 10          /* Maximum number of iterations */ 
#define NUM_BLOCKS 10000    /* Number of blocks to use on the GPU */

typedef struct {            /* 2D vector type */
        double x;
        double y;
} Vector;


__device__ int itr = 0;      /* Iteration count */
int      h_numcenters = 4;   /* Host-side center count */
int      h_numpoints;        /* Host-side point count */
double   h_threshold = 0.05; /* Host-side threshold */
Vector*  h_points;           /* Host-side points */
Vector*  h_centers;          /* Host-side centers */
Vector*  h_tmpcenters;       /* Host-side temporary centers */
int      h_converged;        /* Host-side convergence boolean */
Vector*  d_points;           /* Device-side points */
Vector*  d_centers;          /* Device-side centers */
Vector*  d_tmpcenters;       /* Device-side temporary centers */
int*     d_converged;        /* Device-side convergence boolean */
Vector** d_sums;             /* Device-side cluster sums */
int**    d_counts;           /* Device-side cluster counts */


/*
 * Return a random point
 */
__host__ Vector random_point()
{
        return h_points[rand() % h_numpoints];
}

/*
 * Return a point at (0,0)
 */
__host__ Vector zero_point()
{
        Vector point;
        point.x = 0;
        point.y = 0;
    
        return point;
}

/*
 * Copy the points to the GPU
 */
__host__ void copy_points()
{
        cudaMalloc((void **) &d_points, sizeof(Vector) * h_numpoints);
        cudaMemcpy(d_points, h_points, sizeof(Vector) * h_numpoints,
                   cudaMemcpyHostToDevice);
}

/*
 * Initialize the centers
 */
__host__ void init_centers()
{
        int i;
        for (i = 0; i < h_numcenters; i++) {
                h_centers[i] = random_point();
                h_tmpcenters[i] = zero_point();
        }

        cudaMalloc((void **) &d_centers, sizeof(Vector) * h_numcenters);
        cudaMalloc((void **) &d_tmpcenters, sizeof(Vector) * h_numcenters);
        cudaMemcpy(d_centers, h_centers, sizeof(Vector) * h_numcenters,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_tmpcenters, h_tmpcenters, sizeof(Vector) * h_numcenters,
                   cudaMemcpyHostToDevice);

        /* Initialize the device-side convergence boolean */
        cudaMalloc((void **) &d_converged, sizeof(int));
}

/*
 * Initialize the cluster sums and counts
 */
__host__ void init_sums_counts()
{
        cudaMalloc((void **) &d_sums, sizeof(Vector *) * NUM_BLOCKS);
        cudaMalloc((void **) &d_counts, sizeof(int *) * NUM_BLOCKS);
 
        int i;
        for (i = 0; i < NUM_BLOCKS; i++) {
                cudaMalloc((void **) &d_sums[i], sizeof(Vector) * h_numcenters);
                cudaMalloc((void **) &d_counts[i], sizeof(int) * h_numcenters);
                cudaMemset(d_sums[i], 0, sizeof(Vector) * h_numcenters);
                cudaMemset(d_counts[i], 0, sizeof(int) * h_numcenters);
        }
}

/*
 * Read data points from the input file
 */
__host__ void init_dev(char *inputname)
{
        /* Open the input file */
        if (inputname == NULL) {
                fprintf(stderr, "Must provide an input filename\n");
                free(inputname);
                exit(EXIT_FAILURE);
        }
        FILE *inputfile = fopen(inputname, "r");
        if (inputfile == NULL) {
                fprintf(stderr, "Invalid filename\n");
                free(inputname);
                exit(EXIT_FAILURE);
        }

        /* Read the line count */
        char *line = NULL;
        size_t len = 0;
        ssize_t read = getline(&line, &len, inputfile);
        h_numpoints = atoi(line);

        /* Read each data point in */
        h_points = (Vector *) malloc(sizeof(Vector) * h_numpoints);
        while ((read = getline(&line, &len, inputfile)) != -1) {
                char *saveptr;
                char *token;
                token = strtok_r(line, " ", &saveptr);
                int i = atoi(token) - 1;
        
                token = strtok_r(NULL, " ", &saveptr);
                double x = atof(token);

                token = strtok_r(NULL, " ", &saveptr);
                double y = atof(token);

                h_points[i].x = x;
                h_points[i].y = y;
        }
        h_centers = (Vector *) malloc(sizeof(Vector) * h_numcenters);
        h_tmpcenters = (Vector *) malloc(sizeof(Vector) * h_numcenters);
    
        /* Initialize the data structures on the GPU */
        copy_points();
        init_centers();
        init_sums_counts();

        free(line);
        free(inputname);
        free(h_points);
        free(h_centers);
        free(h_tmpcenters);
        fclose(inputfile);
}

/*
 * Free the cluster sums and counts
 */
__host__ void free_sums_counts()
{
        int i;
        for (i = 0; i < NUM_BLOCKS; i++) {
                cudaFree(d_sums[i]);
                cudaFree(d_counts[i]);
        }

        cudaFree(d_sums);
        cudaFree(d_counts);
} 

/*
 * Free the device resources
 */
__host__ void free_dev()
{
        cudaFree(d_centers);
        cudaFree(d_tmpcenters);
        cudaFree(d_points);
        free_sums_counts();
}

/*
 * Reset the temporary centers, sums, and counts
 */
__device__ void reset_dev(Vector *tmpcenters,
                          Vector **sums,
                          int **counts,
                          int numcenters)
{
        memset(tmpcenters, 0, sizeof(Vector) * numcenters);
        
        int i;
        for (i = 0; i < NUM_BLOCKS; i++) {
                memset(sums[i], 0, sizeof(Vector) * numcenters);
                memset(counts[i], 0, sizeof(int) * numcenters);
        }
}

/*
 * Find the nearest center for each point
 */
__device__ void find_nearest_center(Vector *point,
                                    Vector *centers,
                                    Vector *tmpcenters,
                                    Vector **sums,
                                    int **counts,
                                    int numcenters)
{
        double distance = DBL_MAX;
        int cluster_idx = 0;
        int i;
        for (i = 0; i < numcenters; i++) {
                Vector center = centers[i];
                double d = sqrt(pow(center.x - point->x, 2.0)
                                + pow(center.y - point->y, 2.0));
                if (d < distance) {
                        distance = d;
                        cluster_idx = i;
                } 
        }
        sums[blockIdx.x][cluster_idx].x += point->x;
        sums[blockIdx.x][cluster_idx].y += point->y;
        counts[blockIdx.x][cluster_idx]++;
}

/*
 * Average each cluster and update their centers
 */
__device__ void average_each_cluster(Vector **sums,
                                     int **counts,
                                     int numcenters,
                                     int numpoints)
{ 
        /* Average each cluster and update their centers */
        int cluster_idx;
        for (cluster_idx = 0; cluster_idx < numcenters; cluster_idx++) {
                if (counts[blockIdx.x][cluster_idx] != 0) {
                        Vector sum = sums[blockIdx.x][cluster_idx];
                        int count = counts[blockIdx.x][cluster_idx];
                        double x_avg = sum.x / count;
                        double y_avg = sum.y / count;
                        sums[blockIdx.x][cluster_idx].x = x_avg;
                        sums[blockIdx.x][cluster_idx].y = y_avg;
                }
        }
}

/*
 * Aggregate the centers of each block
 */
__device__ void aggregate_each_block(Vector **sums, Vector *tmpcenters, int numcenters)
{
        int block_idx, cluster_idx;
        for (block_idx = 0; block_idx < NUM_BLOCKS; block_idx++) {
                for (cluster_idx = 0; cluster_idx < numcenters; cluster_idx++) {
                        Vector avg = sums[block_idx][cluster_idx];
                        tmpcenters[cluster_idx].x += (avg.x / NUM_BLOCKS);
                        tmpcenters[cluster_idx].y += (avg.y / NUM_BLOCKS);
                }
        }
}

/*
 * Check if the centers have changed
 */
__device__ int centers_changed(Vector *centers,
                               Vector *tmpcenters,
                               int numcenters,
                               int threshold)
{
        int changed = 0;
        int i;
        for (i = 0; i < numcenters; i++) {
                double x_diff = fabs(tmpcenters[i].x - centers[i].x);
                double y_diff = fabs(tmpcenters[i].y - centers[i].y);
                if (x_diff > threshold || y_diff > threshold)
                        changed = 1;

                centers[i].x = tmpcenters[i].x;
                centers[i].y = tmpcenters[i].y;
        }
    
        return changed;
}

/*
 * Print the results
 */
__device__ void print_results(Vector *centers,
                              int numcenters)
{
        printf("Converged in %d iterations (max=%d)\n", itr, MAX_ITR);

        int i;
        for (i = 0; i < numcenters; i++)
                printf("Cluster %d center: x=%f, y=%f\n", i, centers[i].x, centers[i].y);
}

/*
 * Compute k-means on the device
 */
__global__ void kmeans_kernel(Vector *points,
                              Vector *centers,
                              Vector *tmpcenters,
                              Vector **sums,
                              int **counts,
                              int numcenters,
                              int numpoints,
                              int threshold,
                              int *converged)
{
        int start = blockIdx.x * (numpoints / NUM_BLOCKS);
        int end = (blockIdx.x + 1) * (numpoints / NUM_BLOCKS);
        int cur;
        for (cur = start; cur < end; cur++)
                find_nearest_center(&points[cur], centers, tmpcenters, sums, counts, numcenters);
        average_each_cluster(sums, counts, numcenters, numpoints);

        /* Synchronize here */
        if (blockIdx.x == 0) {
                aggregate_each_block(sums, tmpcenters, numcenters);
                itr++;
                
                *converged = itr >= MAX_ITR || !centers_changed(centers, tmpcenters, numcenters, threshold);
                if (*converged)
                        print_results(centers, numcenters);
                reset_dev(tmpcenters, sums, counts, numcenters);
        }
        /* Synchronize here */
}

/*
 * Host-side wrapper for kmeans_kernel
 */
__host__ void kmeans(char *inputname)
{
        init_dev(inputname);
        do {
                kmeans_kernel<<<NUM_BLOCKS, 1>>>(d_points, d_centers,
                                                 d_tmpcenters, d_sums,
                                                 d_counts, h_numcenters,
                                                 h_numpoints, h_threshold,
                                                 d_converged);
                cudaMemcpy(&h_converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost);
        } while(!h_converged);
        free_dev();
}

int main (int argc,
          char *const *argv)
{
        char* inputname;   
        size_t len;
        int opt;
        while ((opt = getopt(argc, argv, "k:t:i:")) != -1) {
                switch (opt) {
                case 'k':
                        h_numcenters = atoi(optarg);
                        break;
                case 't':
                        h_threshold = atof(optarg);
                        break;
                case 'i':
                        len = strlen(optarg);
                        inputname = (char*) malloc(len + 1);
                        strcpy(inputname, optarg);
                        break;
                default:
                        fprintf(stderr, "Usage: %s [-k clusters] [-t threshold]"
                                " [-i inputfile]\n", argv[0]);
                        exit(EXIT_FAILURE);
                }
        }
        if (inputname == NULL) {
                fprintf(stderr, "Must provide a valid input filename\n");
                exit(EXIT_FAILURE);
        }
    
        kmeans(inputname);
        return 0;
}