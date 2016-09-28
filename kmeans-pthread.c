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
int               _convergence = 0;    /* Boolean for convergence */
int               _itr = 0;            /* Currnet iteration */
int               _max_itr = 10;       /* Maximum allowed iterations */
int               _numpoints;          /* Number of 2D data points */
double            _threshold = 0.05;   /* Threshold for convergence */
pthread_barrier_t _barrierA;           /* Barrier for averaging */
pthread_barrier_t _barrierB;           /* Barrier for convergence */
pthread_barrier_t _barrierC;           /* Barrier for finishing */
char*             _inputname;          /* Input filename to read from */
Vector*           _centers;            /* Cluster centers */
Vector*           _tmpcenters;         /* Temporary cluster centers */
Vector*           _points;             /* 2D data points */
Vector**          _threadsums;         /* 2D array for er-thread cluster sums */
pthread_t*        _threads;            /* pthreads used for averaging */


/*
 * Read data points from the input file
 */
void init_points() {
    /* Open the input file */
    if (_inputname == NULL) {
	fprintf(stderr, "Must provide an input filename\n");
	exit(EXIT_FAILURE);
    }
    
    FILE *inputfile = fopen(_inputname, "r");
    if (inputfile == NULL) {
	fprintf(stderr, "Invalid filename\n");
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

    free(_inputname);
    free(line);
    fclose(inputfile);
}

/*
 * Initialize the array of pthreads
 */
void init_threads() {
    pthread_barrier_init(&_barrierA, NULL, _numthreads);
    pthread_barrier_init(&_barrierB, NULL, _numthreads);
    pthread_barrier_init(&_barrierC, NULL, _numthreads + 1);
    _threads = malloc(sizeof(pthread_t) * _numthreads);

    int i;
    for (i = 0; i < _numthreads; i++) {
	pthread_t thread;
	_threads[i] = thread;
    }
}

/*
 * Return a random center to be associated
 * with a cluster
 */
Vector random_center(int cluster) {
    Vector *point = &_points[rand() % _numpoints];
    point->cluster = cluster;

    return *point;
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
void init_centers() {
    _centers = malloc(sizeof(Vector) * _k);
    _tmpcenters = malloc(sizeof(Vector) * _k);
 
    int i;
    for (i = 0; i < _k; i++) {
	_centers[i] = random_center(i);
	_tmpcenters[i] = zero_center(i);
    }
}

/*
 * Initialize the 2D array for per-thread
 * cluster sums
 */
void init_threadsums() {
    _threadsums = malloc(sizeof(Vector *) * _numthreads);

    int i;
    for (i = 0; i < _numthreads; i++) {
	_threadsums[i] = malloc(sizeof(Vector) * _k);
    }
}

/*
 * Free the array of data points
 */
void free_points() {
    free(_points);
}

/*
 * Free the array of pthreads
 */
void free_threads() {
    free(_threads);
}

/*
 * Free the final array of centers and 
 * the per-iteration array of centers
 */
void free_centers() {
    free(_tmpcenters);
    free(_centers);
}

/*
 * Free the 2D array for per-thread 
 * cluster sums
 */
void free_threadsums() {
    int i;
    for (i = 0; i < _numthreads; i++) {
	free(_threadsums[i]);
    }

    free(_threadsums);
}

/*
 * Reset the per-iteration array
 * of centers
 */
void reset_tmpcenters() {
    int i;
    for (i = 0; i < _k; i++) {
	_tmpcenters[i].x = 0;
	_tmpcenters[i].y = 0;
	_tmpcenters[i].cluster = 0;
    }
}

/*
 * Find the nearest center for each point
 */
void find_nearest_center(Vector *point) {
    double distance = DBL_MAX;
    int cluster_idx = 0;
    int i;
    for (i = 0; i < _k; i++) {
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
void average_each_cluster(int thread_idx) {
    int i;
    int start = thread_idx * (_numpoints / _numthreads);
    int end = (thread_idx + 1) * (_numpoints / _numthreads);
    double xsums[_k];
    double ysums[_k];
    double counts[_k];

    /* Zero out the arrays used for averaging */
    for (i = 0; i < _k; i++) {
	xsums[i] = 0;
	ysums[i] = 0;
	counts[i] = 0;
    }

    /* Sum up each cluster */
    int cur;
    for (cur = start; cur < end; cur++) {
	Vector point = _points[cur];
        xsums[point.cluster] += point.x;
	ysums[point.cluster] += point.y;
	counts[point.cluster] += 1;
    }

    /* Average each cluster and update their centers */
    for (i = 0; i < _k; i++) {
	if (counts[i] == 0) {
	    _threadsums[thread_idx][i].x = 0;
	    _threadsums[thread_idx][i].y = 0;
	} else {
	    double xavg = xsums[i] / counts[i];
	    double yavg = ysums[i] / counts[i];
	    _threadsums[thread_idx][i].x = xavg;
	    _threadsums[thread_idx][i].y = yavg;
	}
    }
}

/*
 * Check if the centers have changed
 */
int centers_changed() {
    int changed = 0;
    int i;
    for (i = 0; i < _k; i++) {
	double x_diff = fabs(_centers[i].x - _tmpcenters[i].x);
	double y_diff = fabs(_centers[i].y - _tmpcenters[i].y);
	if (x_diff > _threshold || y_diff > _threshold) {
	    changed = 1;
	}

	_centers[i].x = _tmpcenters[i].x;
	_centers[i].y = _tmpcenters[i].y;
    }

    return changed;
}

/*
 * Aggregate the centers calculated by 
 * each thread
 */
void aggregate_each_thread() {
    int i, j;
    for (i = 0; i < _numthreads; i++) {
	for (j = 0; j < _k; j++) {
	    _tmpcenters[j].x += (_threadsums[i][j].x / _numthreads);
	    _tmpcenters[j].y += (_threadsums[i][j].y / _numthreads);
	}
    }
}

/*
 * Compute k-means and print out the centers
 */
void *kmeans(void *idx_ptr) {
    int idx = *(int *) idx_ptr;
    int start = idx * (_numpoints / _numthreads);
    int end = (idx + 1) * (_numpoints / _numthreads);
    
    do {
	reset_tmpcenters();
	
	/* Cluster the points and compute the centers */
	int cur;
	for (cur = start; cur < end; cur++) {
	    find_nearest_center(&_points[cur]);
	}
	average_each_cluster(idx);
	
	/* Aggregate the work of each pthread and check for convergence */
	pthread_barrier_wait(&_barrierA);
	if (idx == 0) {
	    aggregate_each_thread();
	    _convergence = !centers_changed() && _itr < _max_itr;
	    _itr++;
	}
	pthread_barrier_wait(&_barrierB);
    } while (!_convergence);

    free(idx_ptr);
    pthread_barrier_wait(&_barrierC);
}

/*
 * Create some pthreads that will each sum up
 * a chunk of the data points
 */
void spawn_worker_threads() {
    int i, s;
    cpu_set_t cpuset;
    
    for (i = 0; i < _numthreads; i++) {
	int *idx_ptr = (int *) malloc(sizeof(int));
	*idx_ptr = i;
	
	s = pthread_create(&_threads[i], NULL, kmeans, idx_ptr);
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
 * After the algorithm has converged, print
 * the centers
 */
void print_centers() {
    printf("Converged in %d iterations (max=%d)\n", _itr, _max_itr);

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
    init_centers();
    init_threadsums();

    spawn_worker_threads();
    pthread_barrier_wait(&_barrierC);
    print_centers();

    free_points();
    free_threads();
    free_centers();
    free_threadsums();
    exit(EXIT_SUCCESS);
}
