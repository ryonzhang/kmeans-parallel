#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include "info.h"


#define MAX_ITR 10            /* Maximum number of iterations */ 
#define THRESHOLD 0.0001      /* Threshold for convergence */

typedef struct {              /* 2D vector type */
        float x;
        float y;
} Vector;


__device__ int itr = 0;       /* Current iteration */
int      _numcenters = 4;     /* Number of centers */
int      _numblocks = 1000;   /* Number of blocks to use on the GPU */
int      _numthreads = 100;   /* Number of per-block threads */
int      _numpoints;          /* Number of points */

Vector*  h_points;            /* Host-side points */
Vector*  h_centers;           /* Host-side centers */
Vector*  h_tmpcenters;        /* Host-side temporary centers */
int*     h_counts;            /* Host-side cluster counts */
int      h_converged;         /* Host-side convergence boolean */

Vector*  d_points;            /* Device-side points */
Vector*  d_centers;           /* Device-side centers */
Vector*  d_tmpcenters;        /* Device-side temporary centers */
int*     d_counts;            /* Device-side cluster counts */
int*     d_converged;         /* Device-side convergence boolean */


/*
 * Return a random point
 */
__host__ Vector random_point()
{
        return h_points[rand() % _numpoints];
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
        cudaMalloc((void **) &d_points, sizeof(Vector) * _numpoints);
        cudaMemcpy(d_points, h_points, sizeof(Vector) * _numpoints,
                   cudaMemcpyHostToDevice);
}

/*
 * Copy the cluster centers and counts to the GPU
 */
__host__ void copy_centers_counts()
{
        cudaMalloc((void **) &d_centers, sizeof(Vector) * _numcenters);
        cudaMalloc((void **) &d_tmpcenters, sizeof(Vector) * _numcenters);
        cudaMalloc((void **) &d_counts, sizeof(int) * _numcenters);
        cudaMemcpy(d_centers, h_centers, sizeof(Vector) * _numcenters,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_tmpcenters, h_tmpcenters, sizeof(Vector) * _numcenters,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_counts, h_counts, sizeof(int) * _numcenters,
                   cudaMemcpyHostToDevice);

        /* Initialize the device-side convergence boolean */
        cudaMalloc((void **) &d_converged, sizeof(int));
}

/*
 * Initialize the data points
 */
__host__ void init_points(char *inputname)
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
        _numpoints = atoi(line);

        /* Read each data point in */
        h_points = (Vector *) malloc(sizeof(Vector) * _numpoints);
        while ((read = getline(&line, &len, inputfile)) != -1) {
                char *saveptr;
                char *token;
                token = strtok_r(line, " ", &saveptr);
                int i = atoi(token) - 1;
        
                token = strtok_r(NULL, " ", &saveptr);
                float x = atof(token);

                token = strtok_r(NULL, " ", &saveptr);
                float y = atof(token);

                h_points[i].x = x;
                h_points[i].y = y;
        }

        /* Copy the points to the GPU */
        copy_points();

        free(line);
        free(inputname);
        fclose(inputfile);
}

/*
 * Initialize the cluster centers and counts
 */
__host__ void init_centers_counts()
{
        h_centers = (Vector *) malloc(sizeof(Vector) * _numcenters);
        h_tmpcenters = (Vector *) malloc(sizeof(Vector) * _numcenters);
        h_counts = (int *) malloc(sizeof(int) * _numcenters);

        /* Initialize the cluster centers and counts on the host */
        int i;
        for (i = 0; i < _numcenters; i++) {
                h_centers[i] = random_point();
                h_tmpcenters[i] = zero_point();
                h_counts[i] = 0;
        }

        /* Copy the cluster centers and counts to the GPU */
        copy_centers_counts();
}

/*
 * Initialize the k-means data structures
 */
__host__ void init_kmeans(char *inputname)
{       
        init_points(inputname);
        init_centers_counts();
}

/*
 * Free the data points
 */
__host__ void free_points()
{
        free(h_points);
        cudaFree(d_points);
}

/*
 * Free the cluster centers and counts
 */
__host__ void free_centers_counts()
{
        free(h_centers);
        free(h_tmpcenters);
        free(h_counts);
        cudaFree(d_centers);
        cudaFree(d_tmpcenters);
        cudaFree(d_counts);
        cudaFree(d_converged);
} 

/*
 * Free the device resources
 */
__host__ void free_kmeans()
{
        free_points();
        free_centers_counts();
}

/*
 * Reset the temporary centers and counts
 */
__device__ void reset_tmpcenters_counts(Vector *tmpcenters,
                                        int *counts,
                                        int numcenters)
{
        memset(tmpcenters, 0, sizeof(Vector) * numcenters);
        memset(counts, 0, sizeof(int) * numcenters);
}

/*
 * Find the nearest center for each point
 */
__device__ void find_nearest_center(Vector *point,
                                    Vector *centers,
                                    Vector *blockcenters,
                                    int *blockcounts,
                                    int numcenters)
{
        float distance = FLT_MAX;
        int cluster_idx = 0;
        int i;
        for (i = 0; i < numcenters; i++) {
                Vector center = centers[i];
                float d = sqrtf(pow(center.x - point->x, 2)
                                + pow(center.y - point->y, 2));
                if (d < distance) {
                        distance = d;
                        cluster_idx = i;
                } 
        }
        atomicAdd(&blockcenters[cluster_idx].x, point->x);
        atomicAdd(&blockcenters[cluster_idx].y, point->y);
        atomicAdd(&blockcounts[cluster_idx], 1);
}

/*
 * Average each cluster and update their centers
 */
__device__ void average_each_cluster(Vector *tmpcenters,
                                     int *counts,
                                     int numcenters)
{ 
        int cluster_idx;
        for (cluster_idx = 0; cluster_idx < numcenters; cluster_idx++) {
                if (counts[cluster_idx] != 0) {
                        Vector sum = tmpcenters[cluster_idx];
                        int count = counts[cluster_idx];
                        float x_avg = sum.x / count;
                        float y_avg = sum.y / count;
                        tmpcenters[cluster_idx].x = x_avg;
                        tmpcenters[cluster_idx].y = y_avg;
                }
        }
}

/*
 * Check if the centers have changed
 */
__device__ int centers_changed(Vector *centers,
                               Vector *tmpcenters,
                               int numcenters)
{
        int changed = 0;
        int i;
        for (i = 0; i < numcenters; i++) {
                float x_diff = fabs(tmpcenters[i].x - centers[i].x);
                float y_diff = fabs(tmpcenters[i].y - centers[i].y);
                if (x_diff > THRESHOLD || y_diff > THRESHOLD)
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
 * Compute k-means across NUM_BLOCKS * NUM_THREADS threads
 */
__global__ void kmeans_work(Vector *points,
                            Vector *centers,
                            Vector *tmpcenters,
                            int *counts,
                            int *converged,
                            int numcenters,
                            int numpoints,
                            int numblocks,
                            int numthreads)
{
	extern __shared__ char shmem[]; /* Shared memory */
	Vector *blockcenters = (Vector *)shmem;
	int *blockcounts = (int *)&blockcenters[numcenters];
	int totalthreads = numblocks * numthreads; /* Total number of threads */

	if (threadIdx.x == 0) {
		memset(blockcenters, 0, sizeof(Vector) * numcenters);
		memset(blockcounts, 0, sizeof(int) * numcenters);
	}
	__syncthreads();

	/* Set the starting index and ending index for each thread */
	int start, end;
	int curthread = (blockIdx.x * numthreads) + threadIdx.x;
	if (curthread > numpoints)
		return;

	if (numpoints < totalthreads) {
		start = curthread;
		end = curthread + 1;
	} else {
		start = curthread * (numpoints / totalthreads); 
		if (numpoints % totalthreads != 0 && curthread == totalthreads - 1)
			end = numpoints;
		else
			end = (curthread + 1) * (numpoints / totalthreads);
	}

	/* Find the nearest center for each point */
	int cur;
        for (cur = start; cur < end; cur++)
                find_nearest_center(&points[cur], centers, blockcenters, blockcounts, numcenters);
	__syncthreads();

	/* Update the global tmpcenters array */
	if (threadIdx.x == 0) {
		int cluster_idx;
		for (cluster_idx = 0; cluster_idx < numcenters; cluster_idx++) {
			atomicAdd(&tmpcenters[cluster_idx].x, blockcenters[cluster_idx].x);
			atomicAdd(&tmpcenters[cluster_idx].y, blockcenters[cluster_idx].y);
			atomicAdd(&counts[cluster_idx], blockcounts[cluster_idx]);
		}
	}
	__syncthreads();
}

/*
 * Average each cluster and check for convergence
 */
__global__ void kmeans_check(Vector *centers,
                             Vector *tmpcenters,
                             int *counts,
                             int *converged,
                             int numcenters)
{
        average_each_cluster(tmpcenters, counts, numcenters);
        itr++;
                
        *converged = itr >= MAX_ITR || !centers_changed(centers, tmpcenters, numcenters);
        if (*converged)
                print_results(centers, numcenters);

        reset_tmpcenters_counts(tmpcenters, counts, numcenters);
}

/*
 * Host-side wrapper for kmeans_kernel
 */
__host__ void kmeans(char *inputname)
{
	unsigned start_kmeans = ticks();
        init_kmeans(inputname);

	int shmem_size = (sizeof(Vector) * _numcenters) + (sizeof(int) * _numcenters);
        unsigned start_gpu = ticks();
        do {
                kmeans_work<<<_numblocks, _numthreads, shmem_size>>>(d_points, d_centers,
								     d_tmpcenters, d_counts,
								     d_converged, _numcenters,
								     _numpoints, _numblocks,
                                                                     _numthreads);
                kmeans_check<<<1, 1>>>(d_centers, d_tmpcenters,
                                       d_counts, d_converged,
                                       _numcenters);
                cudaMemcpy(&h_converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost);
        } while(!h_converged);
        unsigned end_gpu = ticks();
	printf("GPU ticks: %d\n", end_gpu - start_gpu);

        free_kmeans();
	unsigned end_kmeans = ticks();
	printf("End-to-end ticks: %d\n", end_kmeans - start_kmeans);
}

int main (int argc,
          char *const *argv)
{
        /* Parse the input filename */
        if (argc < 2) {
                fprintf(stderr, "Must specify a filename");
                exit(EXIT_FAILURE);
        }
        size_t len = strlen(argv[1]);
        char *inputname = (char *)malloc(len + 1);
        strcpy(inputname, argv[1]);

        /* Parse the options */
        int opt;
        while ((opt = getopt(argc, argv, "k:b:t:")) != -1) {
                switch (opt) {
                case 'k':
                        _numcenters = atoi(optarg);
                        break;
                case 'b':
                        _numblocks = atoi(optarg);
                        break;
                case 't':
                        _numthreads = atoi(optarg);
                        break;
                default:
                        fprintf(stderr, "Not a valid option");
                        exit(EXIT_FAILURE);
                }
        }
    
        kmeans(inputname);
        return 0;
}