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


int     _k = 4;            /* Number of clusters */
double  _threshold = 0.01; /* Threshold for convergence */
char*   _inputfile;        /* Input file to read from */
Vector* _centers;          /* Global array of centers */


/*
 * Return a random point to be associated
 * with a cluster
 */
Vector random_center(int cluster) {
    Vector point;
    point.x = rand() % 100;
    point.y = rand() % 100;
    point.cluster = cluster;

    return point;
}

/*
 * Create the initial, random centers
 */
void init_centers(Vector *points, int num_points,
		   Vector *centers) { 
    int i;
    for (i = 0; i < _k; i++) {
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
 * Average each cluster and update their centers
 */
void average_each_cluster(Vector *centers, Vector *points,
			    int num_points) {
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

    /* Sum up and count each cluster */
    for (i = 0; i < num_points; i++) {
	Vector point = points[i];
	x_sums[point.cluster] += point.x;
	y_sums[point.cluster] += point.y;
	counts[point.cluster] += 1;
    }

    /* Average each cluster and update their centers */
    for (i = 0; i < _k; i++) {
	if (counts[i] != 0) {
	    double x_avg = x_sums[i] / counts[i];
	    double y_avg = y_sums[i] / counts[i];
	    centers[i].x = x_avg;
	    centers[i].y = y_avg;
	} else {
	    centers[i].x = 0;
	    centers[i].y = 0;
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
	double x_diff = abs(centers[i].x - _centers[i].x);
	double y_diff = abs(centers[i].y - _centers[i].y);
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
void kmeans(Vector *points, int num_points) {
    Vector centers[_k];
    init_centers(points, num_points, centers);

    /* While the centers have moved, re-cluster 
	the points and compute the averages */
    int max_itr = 10;
    int itr = 0;
    while (centers_changed(centers) && itr < max_itr) {
	int i;
	for (i = 0; i < num_points; i++) {
	    find_nearest_center(centers, &points[i]);
	}

	average_each_cluster(centers, points, num_points);
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
    char *usage = "Usage: ./kmeans [-k clusters] [-t threshold]"
	          " [-i inputfile]";
    
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
	    _inputfile = (char*) malloc(len + 1);
	    strcpy(_inputfile, optarg);
	    break;
	default:
	    fprintf(stderr, "%s\n", usage);
	    exit(EXIT_FAILURE);
	}
    }

    if (_inputfile != NULL) {
	_centers = malloc(sizeof(Vector) * _k);

	free(_inputfile);
	free(_centers);
    } else {
	fprintf(stderr, "%s\n", usage);
    }
    
    exit(EXIT_SUCCESS);
}
