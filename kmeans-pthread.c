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
} Vector;


int               _k = 4;               /* Number of clusters */
int               _numthreads = 2;      /* Number of pthreads */
int               _convergence = 0;     /* Boolean for convergence */
int               _itr = 0;             /* Currnet iteration */
int               _max_itr = 10;        /* Maximum allowed iterations */
int               _numpoints;           /* Number of 2D data points */
double            _threshold = 0.05;    /* Threshold for convergence */
pthread_barrier_t _barrierA;            /* Barrier for averaging */
pthread_barrier_t _barrierB;            /* Barrier for convergence */
pthread_barrier_t _barrierC;            /* Barrier for finishing */
char*             _inputname;           /* Input filename to read from */
Vector*           _centers;             /* Cluster centers */
Vector*           _tmpcenters;          /* Temporary cluster centers */
Vector*           _points;              /* 2D data points */
Vector**          _partial_sums;        /* 2D array for per-thread cluster sums */
int**             _partial_sums_counts; /* 2D array for per-thread counting of clusters */
pthread_t*        _threads;             /* pthreads used for averaging */


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
 * Return a random vector from the data points
 */
Vector random_vector() {
    return _points[rand() % _numpoints];
}

/*
 * Return a vector at (0, 0)
 */
Vector zero_vector() {
    Vector point;
    point.x = 0;
    point.y = 0;

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
	_centers[i] = random_vector();
	_tmpcenters[i] = zero_vector();
    }
}

/*
 * Initialize the 2D array of per-thread
 * cluster sums
 */
void init_partial_sums() {
    int i, j;

    _partial_sums = (Vector **) malloc(sizeof(Vector *) * _numthreads);
    _partial_sums_counts = (int **) malloc(sizeof(int *) * _numthreads);
    for (i = 0; i < _numthreads; i++) {
	_partial_sums[i] = (Vector *) malloc(sizeof(Vector) * _k);
	_partial_sums_counts[i] = (int *) malloc(sizeof(int) * _k);
    }

    for (i = 0; i < _numthreads; i++) {
	for (j = 0; j < _k; j++) {
	    _partial_sums[i][j].x = 0;
	    _partial_sums[i][j].y = 0;
	    _partial_sums_counts[i][j] = 0;
	}
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
void free_partial_sums() {
    int i;
    for (i = 0; i < _numthreads; i++) {
	free(_partial_sums[i]);
	free(_partial_sums_counts[i]);
    }
    free(_partial_sums);
    free(_partial_sums_counts);
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
    }
}

/* 
 * Reset the 2D array of per-thread
 * cluster sums
 */
void reset_partial_sums() {
    int i, j;
    for (i = 0; i < _numthreads; i++) {
	for (j = 0; j < _k; j++) {
	    _partial_sums[i][j].x = 0;
	    _partial_sums[i][j].y = 0;
	    _partial_sums_counts[i][j] = 0;
	}
    }
}

/*
 * Find the nearest center for each point
 *
 * Return the center's cluster index
 */
int find_nearest_center(Vector *point) {
    double distance = DBL_MAX;
    int cluster_idx = 0;
    int i;
    for (i = 0; i < _k; i++) {
	Vector center = _centers[i];
	double d = sqrt(pow(center.x - point->x, 2.0) + pow(center.y - point->y, 2.0));
	if (d < distance) {
	    distance = d;
	    cluster_idx = i;
	} 
    }

    return cluster_idx;
}

/*
 * Average each cluster and update their centers
 */
void average_each_cluster(int thread_id) {
    int start = thread_id * (_numpoints / _numthreads);
    int end = (thread_id + 1) * (_numpoints / _numthreads);

    /* Average each cluster and update their centers */
    int cluster_idx;
    for (cluster_idx = 0; cluster_idx < _k; cluster_idx++) {
	if (_partial_sums_counts[thread_id][cluster_idx] != 0) {
	    Vector partial_sum = _partial_sums[thread_id][cluster_idx];
	    int count = _partial_sums_counts[thread_id][cluster_idx];
	    double x_avg = partial_sum.x / count;
	    double y_avg = partial_sum.y / count;
	    _partial_sums[thread_id][cluster_idx].x = x_avg;
	    _partial_sums[thread_id][cluster_idx].y = y_avg;
	}
    }
}

/*
 * Aggregate the centers calculated by 
 * each thread
 */
void aggregate_each_thread() {
    int thread_id, cluster_idx;
    for (thread_id = 0; thread_id < _numthreads; thread_id++) {
	for (cluster_idx = 0; cluster_idx < _k; cluster_idx++) {
	    Vector tmpcenter = _partial_sums[thread_id][cluster_idx];
	    _tmpcenters[cluster_idx].x += (tmpcenter.x / _numthreads);
	    _tmpcenters[cluster_idx].y += (tmpcenter.y / _numthreads);
	}
    }
}

/*
 * Check if the centers have changed
 */
int centers_changed() {
    int changed = 0;
    int cluster_idx;
    for (cluster_idx = 0; cluster_idx < _k; cluster_idx++) {
	Vector center = _centers[cluster_idx];
	Vector tmpcenter = _tmpcenters[cluster_idx];
	double x_diff = fabs(center.x - tmpcenter.x);
	double y_diff = fabs(center.y - tmpcenter.y);
	if (x_diff > _threshold || y_diff > _threshold) {
	    changed = 1;
	}

	_centers[cluster_idx].x = _tmpcenters[cluster_idx].x;
	_centers[cluster_idx].y = _tmpcenters[cluster_idx].y;
    }

    return changed;
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

/*
 * Compute k-means and print out the centers
 */
void *kmeans(void *thread_id_ptr) {
    int thread_id = *(int *) thread_id_ptr;
    int start = thread_id * (_numpoints / _numthreads);
    int end = (thread_id + 1) * (_numpoints / _numthreads);
    
    do {
	/* Cluster the points and compute the centers */
	int cur, cluster_idx;
	for (cur = start; cur < end; cur++) {
	    cluster_idx = find_nearest_center(&_points[cur]);
	    _partial_sums[thread_id][cluster_idx].x += _points[cur].x;
	    _partial_sums[thread_id][cluster_idx].y += _points[cur].y;
	    _partial_sums_counts[thread_id][cluster_idx] += 1;
	}
	average_each_cluster(thread_id);

	/* Aggregate the work of each pthread and check for convergence */
	pthread_barrier_wait(&_barrierA);
	if (thread_id == 0) {
	    aggregate_each_thread();
	    _convergence = !centers_changed() && _itr < _max_itr;
	    _itr++;

	    reset_tmpcenters();
	    reset_partial_sums();
	}
	pthread_barrier_wait(&_barrierB);
    } while (!_convergence);

    free(thread_id_ptr);
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
	int *thread_id_ptr = (int *) malloc(sizeof(int));
	*thread_id_ptr = i;
	
	s = pthread_create(&_threads[i], NULL, kmeans, thread_id_ptr);
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
    init_partial_sums();

    spawn_worker_threads();
    pthread_barrier_wait(&_barrierC);
    print_centers();

    free_points();
    free_threads();
    free_centers();
    free_partial_sums();
    exit(EXIT_SUCCESS);
}
