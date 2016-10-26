#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <string.h>


#define MAX_ITR 10             /* Maximum number of iterations */ 
#define NUM_BLOCKS 10000       /* Number of blocks to use on the GPU */

typedef struct {               /* 2D vector type */
        float x;
        float y;
} Vector;


__device__ int itr = 0;        /* Iteration count */
int      _numcenters = 4;     /* Host-side center count */
int      _numpoints;          /* Host-side point count */
float    _threshold = 0.0001; /* Host-side threshold */

Vector*  h_points;             /* Host-side points */
Vector*  h_centers;            /* Host-side centers */
Vector*  h_tmpcenters;         /* Host-side temporary centers */
int*     h_counts;             /* Host-side cluster counts */
int      h_converged;          /* Host-side convergence boolean */

Vector*  d_points;             /* Device-side points */
Vector*  d_centers;            /* Device-side centers */
Vector*  d_tmpcenters;         /* Device-side temporary centers */
int*     d_counts;             /* Device-side cluster counts */
int*     d_converged;          /* Device-side convergence boolean */


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
                                    Vector *tmpcenters,
                                    int *counts,
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
	atomicAdd(&tmpcenters[cluster_idx].x, point->x);
	atomicAdd(&tmpcenters[cluster_idx].y, point->y);
	atomicAdd(&counts[cluster_idx], 1);
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
                               int numcenters,
                               float threshold)
{
        int changed = 0;
        int i;
        for (i = 0; i < numcenters; i++) {
                float x_diff = fabs(tmpcenters[i].x - centers[i].x);
                float y_diff = fabs(tmpcenters[i].y - centers[i].y);
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
 * Compute k-means across NUM_BLOCKS blocks
 */
__global__ void kmeans_work(Vector *points,
                            Vector *centers,
                            Vector *tmpcenters,
                            int *counts,
                            int *converged,
                            int numcenters,
                            int numpoints,
                            float threshold)
{
        int start = blockIdx.x * (numpoints / NUM_BLOCKS);
        int end = (blockIdx.x + 1) * (numpoints / NUM_BLOCKS);
        int cur;
        for (cur = start; cur < end; cur++)
                find_nearest_center(&points[cur], centers, tmpcenters, counts, numcenters);
}

/*
 * Average each cluster and check for convergence
 */
__global__ void kmeans_check(Vector *centers,
			     Vector *tmpcenters,
			     int *counts,
			     int *converged,
			     int numcenters,
			     float threshold)
{ 
	average_each_cluster(tmpcenters, counts, numcenters);
	itr++;
                
	*converged = itr >= MAX_ITR || !centers_changed(centers, tmpcenters, numcenters, threshold);
	if (*converged)
		print_results(centers, numcenters);

	reset_tmpcenters_counts(tmpcenters, counts, numcenters);
}

/*
 * Host-side wrapper for kmeans_kernel
 */
__host__ void kmeans(char *inputname)
{
        init_kmeans(inputname);

        do {
                kmeans_work<<<NUM_BLOCKS, 1>>>(d_points, d_centers,
                                               d_tmpcenters, d_counts,
					       d_converged, _numcenters,
					       _numpoints, _threshold);
                kmeans_check<<<1, 1>>>(d_centers, d_tmpcenters,
				       d_counts, d_converged,
				       _numcenters, _threshold);
                cudaDeviceSynchronize();
                cudaMemcpy(&h_converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost);
        } while(!h_converged);
        
	free_kmeans();
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
                        _numcenters = atoi(optarg);
                        break;
                case 't':
                        _threshold = atof(optarg);
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