#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <string.h>


typedef struct {    /* A 2D vector */
    double x;
    double y;
    int cluster;
} Vector;


int         _numthreads = 2;   /* Number of pthreads */
int         _k = 4;            /* Number of clusters */
double      _threshold = 0.05; /* Threshold for convergence */
char*       _inputname;        /* Input filename to read from */
Vector*     _centers;          /* Global array of centers */
Vector*     _points;           /* Global array of 2D data points */
pthread_t*  _threads;          /* Global array of pthreads */
int         _numpoints;        /* Number of 2D data points */


/*
 * Initialize the array of pthreads */
void init_threads() {
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
void init_points(char *inputname) {
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
void *sum_chunk(void *arg) {
    /*
    int i;
    for (i = start; i < end; i++) {
	Vector point = _points[i];
	x_sums[point.cluster] += point.x;
	y_sums[point.cluster] += point.y;
	counts[point.cluster] += 1;
    }
    */

    pthread_exit(NULL);
}

/*
 * Average each cluster and update their centers
 */
void average_each_cluster(Vector *centers) {
    /* Initialize the arrays */
    double x_sums[_k];
    double y_sums[_k];
    int counts[_k];
    int i;

    for (i = 0; i < _k; i++) {
	x_sums[i] = 0;
	y_sums[i] = 0;
	counts[i] = 0;
    }

    /* Create some pthreads */
    for (i = 0; i < _numthreads; i++) {
	int success = pthread_create(&_threads[i], NULL,
				     sum_chunk, NULL);
	
	if (success != 0) {
	    fprintf(stderr, "Couldn't create a pthread\n");
	    exit(EXIT_FAILURE);
	}
    }

    /* Wait for the pthreads to exit */
    for (i = 0; i < _numthreads; i++) {
	int success = pthread_join(_threads[i], NULL);
	
	if (success != 0) {
	    fprintf(stderr, "Couldn't join a pthread\n");
	    exit(EXIT_FAILURE);
	}
    }

    /* Average each cluster and update their centers */
    for (i = 0; i < _k; i++) {
	if (counts[i] == 0) {
	    centers[i].x = 0;
	    centers[i].y = 0;
	} else {
	    double x_avg = x_sums[i] / counts[i];
	    double y_avg = y_sums[i] / counts[i];
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
    while ((opt = getopt(argc, argv, "k:t:i:")) != -1) {
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
    kmeans();

    free(_inputname);
    free(_centers);
    free(_points);
    free(_threads);    
    exit(EXIT_SUCCESS);
}
