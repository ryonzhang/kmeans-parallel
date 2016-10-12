#include <iostream>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <string.h>


typedef struct {    /* A 2D vector */
    double x;
    double y;
    int cluster;
} Vector;


__device__ int     _k = 4;            /* Number of clusters */
__device__ double  _threshold = 0.05; /* Threshold for convergence */
__device__ int     _numpoints;        /* Number of 2D data points */
__device__ Vector* _points;           /* Global array of 2D data points */
__device__ Vector* _centers;
__device__ Vector* _tmpcenters;


/*
 * Return a random point
 */
__host__ Vector random_point(Vector *points, int numpoints) {
    return points[rand() % numpoints];
}

/*
 * Return a point at (0,0)
 */
__host__ Vector zero_point() {
    Vector point;
    point.x = 0;
    point.y = 0;

    return point;
}

/*
 * Copy the points to the GPU
 */
__host__ void init_points(Vector *points, int numpoints) {
    cudaMalloc((Vector **) &_points, sizeof(Vector) * numpoints);
    cudaMemcpy(&points, _points, sizeof(Vector) * numpoints, cudaMemcpyHostToDevice);
}

/*
 * Copy the initial centers to the GPU
 */
__host__ void init_centers(int k, Vector *points, int numpoints) {
    Vector centers[k];
    Vector tmpcenters[k];
    int i;
    for (i = 0; i < k; i++) {
	centers[i] = zero_point();
	tmpcenters[i] = random_point(points, numpoints);
    }

    /* Copy the centers to the GPU */
    cudaMalloc((Vector **) &_centers, sizeof(Vector) * k);
    cudaMalloc((Vector **) &_tmpcenters, sizeof(Vector) * k);
    cudaMemcpy(&centers, _centers, sizeof(Vector) * k, cudaMemcpyHostToDevice);
    cudaMemcpy(&tmpcenters, _tmpcenters, sizeof(Vector) * k, cudaMemcpyHostToDevice);
}

/*
 * Read data points from the input file
 */
__host__ void init_dev(char *inputname) {
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
    int numpoints = atoi(line);
    Vector points[numpoints];

    /* Read each data point in */
    while ((read = getline(&line, &len, inputfile)) != -1) {
	char *saveptr;
	char *token;
	token = strtok_r(line, " ", &saveptr);
	int i = atoi(token) - 1;
	
	token = strtok_r(NULL, " ", &saveptr);
	double x = atof(token);

	token = strtok_r(NULL, " ", &saveptr);
	double y = atof(token);

	points[i].x = x;
	points[i].y = y;
    }

    /* Initialize the data structures on the GPU */
    init_points(points, numpoints);
    init_centers(k, points, numpoints);
    
    free(line);
    free(inputname);
    fclose(inputfile);
}

__host__ void free_dev() {
    cudaFree(_centers);
    cudaFree(_tmpcenters);
    cudaFree(_points);
}

/*
 * Find the nearest center for each point
 */
__device__ void find_nearest_center(Vector *point) {
    double distance = DBL_MAX;
    int cluster_idx = 0;
    int i;
    for (i = 0; i < k; i++) {
	Vector center = _centers[i];
	double d = sqrt(pow(center.x - point->x, 2.0)
			       + pow(center.y - point->y, 2.0));
	if (d < distance) {
	    distance = d;
	    cluster_idx = i;
	} 
    }

    point->cluster = cluster_idx;
}

/*
 * Average each cluster and update their centers
 */
__device__ void average_each_cluster() {
    /* Initialize the arrays */
    double *x_sums = (double *) malloc(sizeof(double) * k);
    double *y_sums = (double *) malloc(sizeof(double) * k);
    int *counts = (int *) malloc(sizeof(int) * k);
    int i;
    for (i = 0; i < k; i++) {
	x_sums[i] = 0;
	y_sums[i] = 0;
	counts[i] = 0;
    }

    /* Sum up and count each cluster */
    for (i = 0; i < numpoints; i++) {
	Vector point = points[i];
	x_sums[point.cluster] += point.x;
	y_sums[point.cluster] += point.y;
	counts[point.cluster] += 1;
    }

    /* Average each cluster and update their centers */
    for (i = 0; i < k; i++) {
	if (counts[i] != 0) {
	    double x_avg = x_sums[i] / counts[i];
	    double y_avg = y_sums[i] / counts[i];
	    _tmpcenters[i].x = x_avg;
	    _tmpcenters[i].y = y_avg;
	} else {
	    _tmpcenters[i].x = 0;
	    _tmpcenters[i].y = 0;
	}
    }

    free(x_sums);
    free(y_sums);
    free(counts);
}

/*
 * Check if the centers have changed
 */
__device__ int centers_changed() {
    int changed = 0;
    int i;
    for (i = 0; i < _k; i++) {
	double x_diff = fabs(_tmpcenters[i].x - _centers[i].x);
	double y_diff = fabs(_tmpcenters[i].y - _centers[i].y);
	if (x_diff > threshold || y_diff > threshold) {
	    changed = 1;
	}

	_centers[i].x = _tmpcenters[i].x;
	_centers[i].y = _tmpcenters[i].y;
    }

    return changed;
}

/*
 * Compute k-means on the GPU and print out the centers
 */
__global__ void kmeans() {
    /* While the centers have moved, re-cluster 
	the points and compute the averages */
    int itr, i = 0;
    int max_itr = 10;
    do {
	int i;
	for (i = 0; i < numpoints; i++) {
	    find_nearest_center(&_points[i]);
	}

	average_each_cluster();
	itr++;
    } while (centers_changed() && itr < max_itr);
    printf("Converged in %d iterations (max=%d)\n", itr, max_itr);
    
    /* Print the center of each cluster */
    for (i = 0; i < _k; i++) {
	printf("Cluster %d center: x=%f, y=%f\n",
	       i, _centers[i].x, _centers[i].y);
    }
}

int main (int argc, char *const *argv) {
    char* inputname;
    size_t len;
    int opt;
    while ((opt = getopt(argc, argv, "k:t:i:")) != -1) {
	switch (opt) {
	case 'k':
	    _k = atoi(optarg);
	    break;
	case 't':
	    _threshold = atof(optarg);
	    break;
	case 'i':
	    len = strlen(optarg);
	    inputname = (char*) malloc(len + 1);
	    strcpy(_inputname, optarg);
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
    
    init_dev(inputname);
    kmeans<<<1, 1>>>();
    cudaError_t cudaerr = cudaDeviceSynchronize();
    free_dev();
    
    if (cudaerr != cudaSuccess) {
	fprintf(stderr, "Kernel launch failed with error \"%s\". \n", cudaGetErrorString(cudaerr));
	exit(EXIT_FAILURE);
    }
    return 0;
}