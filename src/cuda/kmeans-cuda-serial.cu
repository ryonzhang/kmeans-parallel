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

typedef struct {            /* 2D vector type */
    double x;
    double y;
} Vector;


__device__ int itr = 0;     /* Iteration count */
int     h_numcenters = 4;   /* Host-side center count */
int     h_numpoints;        /* Host-side point count */
double  h_threshold = 0.05; /* Host-side threshold */
Vector* h_centers;          /* Host-side centers */
Vector* h_tmpcenters;       /* Host-side temporary centers */
Vector* h_points;           /* Host-side points */
int*    h_counts;           /* Host-side cluster counts */
int     h_converged;        /* Host-side convergence boolean */
Vector* d_centers;          /* Device-side centers */
Vector* d_tmpcenters;       /* Device-side temporary centers */
Vector* d_points;           /* Device-side points */
int*    d_counts;           /* Device-side cluster counts */
int*    d_converged;        /* Device-side convergence boolean */

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
__host__ void init_points()
{
    cudaMalloc((void **) &d_points, sizeof(Vector) * h_numpoints);
    cudaMemcpy(d_points, h_points, sizeof(Vector) * h_numpoints,
               cudaMemcpyHostToDevice);
}

/*
 * Copy the initial centers to the GPU
 */
__host__ void init_centers()
{
    int i;
    for (i = 0; i < h_numcenters; i++) {
        h_centers[i] = random_point();
        h_tmpcenters[i] = zero_point();
        h_counts[i] = 0;
    }

    /* Copy the centers to the GPU */
    cudaMalloc((void **) &d_centers, sizeof(Vector) * h_numcenters);
    cudaMalloc((void **) &d_tmpcenters, sizeof(Vector) * h_numcenters);
    cudaMalloc((void **) &d_counts, sizeof(int) * h_numcenters);
    cudaMemcpy(d_centers, h_centers, sizeof(Vector) * h_numcenters,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmpcenters, h_tmpcenters, sizeof(Vector) * h_numcenters,
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_counts, h_counts, sizeof(int) * h_numcenters,
               cudaMemcpyHostToDevice);

    /* Initialize the device-side convergence boolean */
    cudaMalloc((void **) &d_converged, sizeof(int));
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
    h_counts = (int *) malloc(sizeof(Vector) * h_numcenters);
    
    /* Initialize the data structures on the GPU */
    init_points();
    init_centers();

    free(line);
    free(inputname);
    free(h_points);
    free(h_centers);
    free(h_tmpcenters);
    free(h_counts);
    fclose(inputfile);
}

/*
 * Reset the temporary centers and counts
 */
__device__ void reset_tmpcenters(Vector *tmpcenters,
                                 int *counts,
                                 int numcenters)
{
    int i;
    for (i = 0; i < numcenters; i++) {
        tmpcenters[i].x = 0;
        tmpcenters[i].y = 0;
        counts[i] = 0;
    }
}

/*
 * Free the device resources
 */
__host__ void free_dev()
{
    cudaFree(d_centers);
    cudaFree(d_tmpcenters);
    cudaFree(d_points);
}

/*
 * Find the nearest center for each point
 */
__device__ int find_nearest_center(Vector *point,
                                   Vector *centers,
                                   Vector *tmpcenters,
                                   int *counts,
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
    tmpcenters[cluster_idx].x += point->x;
    tmpcenters[cluster_idx].y += point->y;
    counts[cluster_idx]++;

    return cluster_idx;
}

/*
 * Average each cluster and update their centers
 */
__device__ void average_each_cluster(Vector *tmpcenters,
                                     int *counts,
                                     int numcenters,
                                     Vector *points,
                                     int numpoints)
{
    /* Average each cluster and update their centers */
    int i;
    for (i = 0; i < numcenters; i++) {
        if (counts[i] != 0) {
            double x_avg = tmpcenters[i].x / counts[i];
            double y_avg = tmpcenters[i].y / counts[i];
            tmpcenters[i].x = x_avg;
            tmpcenters[i].y = y_avg;
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
			      int *counts,
			      int numcenters,
			      int numpoints,
			      int threshold,
			      int *converged)
{
    /* Re-cluster the points, compute the averages,
     * and check for convergence */
    reset_tmpcenters(tmpcenters, counts, numcenters);
    int i;
    for (i = 0; i < numpoints; i++)
	find_nearest_center(&points[i], centers, tmpcenters, counts, numcenters);
    average_each_cluster(tmpcenters, counts, numcenters, points, numpoints);

    itr++;
    *converged = itr >= MAX_ITR || !centers_changed(centers, tmpcenters, numcenters, threshold);
    if (*converged)
	print_results(centers, numcenters);
}

/*
 * Host-side wrapper for kmeans_kernel
 */
__host__ void kmeans(char *inputname)
{
    init_dev(inputname);
    do {
	kmeans_kernel<<<1, 1>>>(d_points, d_centers, d_tmpcenters, d_counts,
				h_numcenters, h_numpoints, h_threshold,
				d_converged);
	cudaMemcpy(&h_converged, d_converged, sizeof(int), cudaMemcpyDeviceToHost);
    } while(!h_converged);

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed with error \"%s\". \n",
                cudaGetErrorString(cudaerr));
    }
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