all: kmeans

kmeans: kmeans.c
	gcc kmeans.c -lm -o kmeans

clean:
	rm kmeans

