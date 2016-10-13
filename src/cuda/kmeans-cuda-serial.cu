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


int     k = 4;
int     numpoints;
double  threshold = 0.05;
Vector* h_centers;
Vector* h_tmpcenters;
Vector* h_points; 
Vector* d_centers;
Vector* d_tmpcenters;
Vector* d_points;


/*
 * Return a random point
 */
__host__ Vector random_point() {
    return h_points[rand() % numpoints];
}

/*
 * Return a point at (0,0)
 */
__host__ Vector zero_point() {
    Vector point;
    point.x = 0;
    point.y = 0;
    point.cluster = 0;

    return point;
}

/*
 * Copy the points to the GPU
 */
__host__ void init_points() {
    cudaMalloc((void **) &d_points, sizeof(Vector) * numpoints);
    cudaMemcpy(d_points, h_points, sizeof(Vector) * numpoints, cudaMemcpyHostToDevice);
}

/*
 * Copy the initial centers to the GPU
 */
__host__ void init_centers() {
    int i;
    for (i = 0; i < k; i++) {
	h_centers[i] = zero_point();
	h_tmpcenters[i] = random_point();
    }

    /* Copy the centers to the GPU */
    cudaMalloc((void **) &d_centers, sizeof(Vector) * k);
    cudaMalloc((void **) &d_tmpcenters, sizeof(Vector) * k);
    cudaMemcpy(d_centers, h_centers, sizeof(Vector) * k, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmpcenters, h_tmpcenters, sizeof(Vector) * k, cudaMemcpyHostToDevice);
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
    numpoints = atoi(line);

    /* Read each data point in */
    h_points = (Vector *) malloc(sizeof(Vector) * numpoints);
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
    h_centers = (Vector *) malloc(sizeof(Vector) * k);
    h_tmpcenters = (Vector *) malloc(sizeof(Vector) * k);
    
    /* Initialize the data structures on the GPU */
    init_points();
    init_centers();

    free(line);
    free(inputname);
    free(h_points);
    free(h_centers);
    free(h_tmpcenters);
    fclose(inputfile);
}

__host__ void free_dev() {
    cudaFree(d_centers);
    cudaFree(d_tmpcenters);
    cudaFree(d_points);
}

/*
 * Find the nearest center for each point
 */
__device__ void find_nearest_center(Vector *point, Vector *_d_tmpcenters, int _k) {
    double distance = DBL_MAX;
    int cluster_idx = 0;
    int i;
    for (i = 0; i < _k; i++) {
	Vector center = _d_tmpcenters[i];
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
__device__ void average_each_cluster(Vector *_d_tmpcenters, int _k, Vector *_d_points, int _numpoints) {
    /* Initialize the arrays */
    double *x_sums = (double *) malloc(sizeof(double) * _k);
    double *y_sums = (double *) malloc(sizeof(double) * _k);
    int *counts = (int *) malloc(sizeof(int) * _k);
    int i;
    for (i = 0; i < _k; i++) {
	x_sums[i] = 0;
	y_sums[i] = 0;
	counts[i] = 0;
    }

    /* Sum up and count each cluster */
    for (i = 0; i < _numpoints; i++) {
	Vector point = _d_points[i];
	x_sums[point.cluster] += point.x;
	y_sums[point.cluster] += point.y;
	counts[point.cluster] += 1;
    }

    /* Average each cluster and update their centers */
    for (i = 0; i < _k; i++) {
	if (counts[i] != 0) {
	    double x_avg = x_sums[i] / counts[i];
	    double y_avg = y_sums[i] / counts[i];
	    _d_tmpcenters[i].x = x_avg;
	    _d_tmpcenters[i].y = y_avg;
	} else {
	    _d_tmpcenters[i].x = 0;
	    _d_tmpcenters[i].y = 0;
	}
    }

    free(x_sums);
    free(y_sums);
    free(counts);
}

/*
 * Check if the centers have changed
 */
__device__ int centers_changed(Vector *_d_centers, Vector *_d_tmpcenters, int _k,
			       int _threshold) {
    int changed = 0;
    int i;
    for (i = 0; i < _k; i++) {
	double x_diff = fabs(_d_tmpcenters[i].x - _d_centers[i].x);
	double y_diff = fabs(_d_tmpcenters[i].y - _d_centers[i].y);
	if (x_diff > _threshold || y_diff > _threshold) {
	    changed = 1;
	}

	_d_centers[i].x = _d_tmpcenters[i].x;
	_d_centers[i].y = _d_tmpcenters[i].y;
    }

    return changed;
}

/*
 * Compute k-means on the GPU and print out the centers
 */
__global__ void kmeans(Vector *_d_points, Vector *_d_centers, Vector *_d_tmpcenters,
		       int _k, int _numpoints, int _threshold) {
    /* While the centers have moved, re-cluster 
	the points and compute the averages */
    int itr, i = 0;
    int max_itr = 10;
    do {
	int i;
	for (i = 0; i < _numpoints; i++) {
	    find_nearest_center(&_d_points[i], _d_tmpcenters, _k);
	}

	average_each_cluster(_d_tmpcenters, _k, _d_points, _numpoints);
	itr++;
    } while (centers_changed(_d_centers, _d_tmpcenters, _k, _threshold)
	     && itr < max_itr);
    printf("Converged in %d iterations (max=%d)\n", itr, max_itr);
    
    /* Print the center of each cluster */
    for (i = 0; i < _k; i++) {
	printf("Cluster %d center: x=%f, y=%f\n",
	       i, _d_centers[i].x, _d_centers[i].y);
    }
}

int main (int argc, char *const *argv) {
    int k;
    double threshold;
    char* inputname;
    
    size_t len;
    int opt;
    while ((opt = getopt(argc, argv, "k:t:i:")) != -1) {
	switch (opt) {
	case 'k':
	    k = atoi(optarg);
	    break;
	case 't':
	    threshold = atof(optarg);
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
    
    init_dev(inputname);
    kmeans<<<1, 1>>>(d_points, d_centers, d_tmpcenters,
		     k, numpoints, threshold);
    
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess) {
	fprintf(stderr, "Kernel launch failed with error \"%s\". \n", cudaGetErrorString(cudaerr));
    }
    
    free_dev();
    return 0;
}