all: kmeans-serial kmeans-fork

kmeans-serial: kmeans-serial.c
	gcc kmeans-serial.c -lm -o kmeans-serial

kmeans-fork: kmeans-fork.c
	gcc kmeans-fork.c -lm -o kmeans-fork

clean:
	rm kmeans-serial
	rm kmeans-fork

