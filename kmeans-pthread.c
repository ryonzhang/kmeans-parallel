#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <string.h>


typedef struct {    /* A 2D vector */
    double x;
    double y;
    int cluster;
} Vector;


int               _k = 4;              /* Number of clusters */
int               _numthreads = 2;     /* Number of pthreads */
int               _numpoints;          /* Number of 2D data points */
double            _threshold = 0.05;   /* Threshold for convergence */
pthread_barrier_t _barrier;            /* Barrier for the pthreads */
char*             _inputname;          /* Input filename to read from */
Vector*           _centers;            /* Cluster centers */
Vector*           _points;             /* 2D data points */
pthread_t*        _threads;            /* pthreads used for averaging */
double*           _xsums;              /* x-axis sum for each cluster */
double*           _ysums;              /* y-axis sum for each cluster */
int*              _counts;             /* Count for each cluster */


/*
 * Initialize the array of pthreads
 */
void init_threads() {
    pthread_barrier_init(&_barrier, NULL, _numthreads + 1);
    _threads = malloc(sizeof(pthread_t) * _numthreads);

    int i;
    for (i = 0; i < _numthreads; i++) {
	pthread_t thread;
	_threads[i] = thread;
    }

}

/*
 * Read data points from the input file
 */
void init_points() {
    _centers = malloc(sizeof(Vector) * _k);
    
    /* Open the input file */
    if (_inputname == NULL) {
	fprintf(stderr, "Must provide an input filename\n");
	free(_inputname);
	free(_centers);
	exit(EXIT_FAILURE);
    }
    
    FILE *inputfile = fopen(_inputname, "r");
    if (inputfile == NULL) {
	fprintf(stderr, "Invalid filename\n");
	free(_inputname);
	free(_centers);
	exit(EXIT_FAILURE);
    }

    /* Read the line count */
    char *line = NULL;
    size_t len = 0;
    ssize_t read = getline(&line, &len, inputfile);
    _numpoints = atoi(line);
    _points = malloc(sizeof(Vector) * _numpoints);

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

	_points[i].x = x;
	_points[i].y = y;
	_points[i].cluster = 0;
    }

    free(line);
    fclose(inputfile);
}

/*
 * Initialize the arrays used to average 
 * the data points
 */
void init_avg_arrays() {
    _xsums = malloc(sizeof(double) * _k);
    _ysums = malloc(sizeof(double) * _k);
    _counts = malloc(sizeof(int) * _k);
}

/*
 * Return a random center to be associated
 * with a cluster
 */
Vector random_center(int cluster) {
    return _points[rand() % _numpoints];
}

/*
 * Return a center at (0,0) to be associated
 * with a cluster
 */
Vector zero_center(int cluster) {
    Vector point;
    point.x = 0;
    point.y = 0;
    point.cluster = cluster;

    return point;
}

/*
 * Create the initial, random centers
 */
void init_centers(Vector *centers) { 
    int i;
    for (i = 0; i < _k; i++) {
	_centers[i] = zero_center(i);
	centers[i] = random_center(i);
    }
}

/*
 * Find the nearest center for each point
 */
void find_nearest_center(Vector *centers, Vector *point) {
    double distance = DBL_MAX;
    int cluster_idx = 0;
    int i;
    for (i = 0; i < _k; i++) {
	Vector center = centers[i];
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
 * Sum up a chunk of the points and update 
 * the cluster counts
 */
void *sum_chunk(void *idx_ptr) {
    int i = *(int *) idx_ptr;
    int start = i * (_numpoints / _numthreads);
    int end = (i + 1) * (_numpoints / _numthreads);
    
    int cur;
    for (cur = start; cur < end; cur++) {
	Vector point = _points[cur];
        _xsums[point.cluster] += point.x;
	_ysums[point.cluster] += point.y;
	_counts[point.cluster] += 1;
    }

    free(idx_ptr);
    pthread_barrier_wait(&_barrier);
}

/*
 * Create some pthreads that will each sum up
 * a chunk of the data points
 */
void create_chunk_threads() {
    int i, s;
    cpu_set_t cpuset;
    
    for (i = 0; i < _numthreads; i++) {
	int *idx_ptr = (int *) malloc(sizeof(int));
	*idx_ptr = i;
	
	s = pthread_create(&_threads[i], NULL, sum_chunk, idx_ptr);
	if (s != 0) {
	    fprintf(stderr, "Couldn't create a pthread\n");
	    exit(EXIT_FAILURE);
	}

	CPU_ZERO(&cpuset);
	CPU_SET(i, &cpuset);
	s = pthread_setaffinity_np(_threads[i], sizeof(cpu_set_t), &cpuset);
	if (s != 0) {
	    fprintf(stderr, "Couldn't set the affinity of a pthread\n");
	    exit(EXIT_FAILURE);
	}
    }
}

/*
 * Average each cluster and update their centers
 */
void average_each_cluster(Vector *centers) {
    /* Zero out the arrays used for averaging */
    int i;
    for (i = 0; i < _k; i++) {
	_xsums[i] = 0;
	_ysums[i] = 0;
	_counts[i] = 0;
    }

    /* Create some pthreads to sum up the points */
    create_chunk_threads();
    pthread_barrier_wait(&_barrier);

    /* Average each cluster and update their centers */
    for (i = 0; i < _k; i++) {
	if (_counts[i] == 0) {
	    centers[i].x = 0;
	    centers[i].y = 0;
	} else {
	    double x_avg = _xsums[i] / _counts[i];
	    double y_avg = _ysums[i] / _counts[i];
	    centers[i].x = x_avg;
	    centers[i].y = y_avg;
	}
    }
}

/*
 * Check if the centers have changed
 */
int centers_changed(Vector *centers) {
    int changed = 0;
    int i;
    for (i = 0; i < _k; i++) {
	double x_diff = fabs(centers[i].x - _centers[i].x);
	double y_diff = fabs(centers[i].y - _centers[i].y);
	if (x_diff > _threshold || y_diff > _threshold) {
	    changed = 1;
	}

	_centers[i].x = centers[i].x;
	_centers[i].y = centers[i].y;
    }

    return changed;
}

/*
 * Compute k-means and print out the centers
 */
void kmeans() {
    Vector centers[_k];
    init_centers(centers);
    
    /* While the centers have moved, re-cluster 
	the points and compute the averages */
    int max_itr = 10;
    int itr = 0;
    while (centers_changed(centers) && itr < max_itr) {
	int i;
	for (i = 0; i < _numpoints; i++) {
	    find_nearest_center(centers, &_points[i]);
	}
	average_each_cluster(centers);
	itr++;
    }
    printf("Converged in %d iterations (max=%d)\n", itr, max_itr);
    
    /* Print the center of each cluster */
    int j;
    for (j = 0; j < _k; j++) {
	printf("Cluster %d center: x=%f, y=%f\n",
	       j, _centers[j].x, _centers[j].y);
    }
}

void main (int argc, char *const *argv) {
    size_t len;
    int opt;
    while ((opt = getopt(argc, argv, "k:t:p:i:")) != -1) {
	switch (opt) {
	case 'k':
	    _k = atoi(optarg);
	    break;
	case 't':
	    _threshold = atof(optarg);
	    break;
	case 'p':
	    _numthreads = atoi(optarg);
	    break;
	case 'i':
	    len = strlen(optarg);
	    _inputname = (char*) malloc(len + 1);
	    strcpy(_inputname, optarg);
	    break;
	default:
	    fprintf(stderr, "Usage: %s [-k clusters] [-t threshold]"
                            " [-i inputfile]\n", argv[0]);
	    exit(EXIT_FAILURE);
	}
    }

    init_points();
    init_threads();
    init_avg_arrays();

    kmeans();

    free(_inputname);
    free(_centers);
    free(_points);
    free(_threads);
    free(_xsums);
    free(_ysums);
    free(_counts);
    exit(EXIT_SUCCESS);
}
