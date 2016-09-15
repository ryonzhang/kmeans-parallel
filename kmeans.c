#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>

typedef struct {  /* A 2D vector */
    double x;
    double y;
    int cluster;
} Vector;


Vector *_centers; /* Global array of centers */


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
void init_centers(int k, Vector *points, int num_points,
		    Vector *centers) { 
    int i;
    for (i = 0; i < k; i++) {
	centers[i] = random_center(i);
    }
}

/*
 * Find the nearest center for each point
 */
void find_nearest_center(Vector *centers, int k, Vector *point) {
    double distance = DBL_MAX;
    int cluster_idx = 0;
    int i;
    for (i = 0; i < k; i++) {
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
void average_each_cluster(Vector *centers, int k, Vector *points,
			  int num_points) {
    /* Initialize the arrays */
    double x_sums[k];
    double y_sums[k];
    int counts[k];
    int i;
    for (i = 0; i < k; i++) {
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
    for (i = 0; i < k; i++) {
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
 * 
 * Chris' suggestion: Add a margin of error
 * when comparing centers
 */
int centers_changed(Vector *centers, int k) {
    int changed = 0;
    double epsilon = 0.001;
    int i;
    for (i = 0; i < k; i++) {
	double x_diff = abs(centers[i].x - _centers[i].x);
	double y_diff = abs(centers[i].y - _centers[i].y);
	if (x_diff > epsilon || y_diff > epsilon) {
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
void kmeans(int k, Vector *points, int num_points) {
    Vector centers[k];
    init_centers(k, points, num_points, centers);

    /* While the centers have moved, re-cluster 
	the points and compute the averages */
    int max_itr = 10;
    int itr = 0;
    while (centers_changed(centers, k) && itr < max_itr) {
	int i;
	for (i = 0; i < num_points; i++) {
	    find_nearest_center(centers, k, &points[i]);
	}

	average_each_cluster(centers, k, points, num_points);
	itr++;
    }
    printf("Converged in %d iterations (max=%d)\n", itr, max_itr);
    
    /* Print the center of each cluster */
    int j;
    for (j = 0; j < k; j++) {
	printf("Cluster %d center: x=%f, y=%f\n",
	       j, _centers[j].x, _centers[j].y);
    }
}

void main (int argc, char** argv) {
    _centers = malloc(sizeof(Vector) * 4);
    Vector points[100];

    int i;
    for (i = 0; i < 100; i++) {
	Vector v;
	v.x = (double) (rand() % 100);
	v.y = (double) (rand() % 100);
	v.cluster = 0;
	points[i] = v;
    }
    kmeans(4, points, 100);
    
    free((void*) _centers);
}
