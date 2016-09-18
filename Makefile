all: kmeans-serial kmeans-pthread

kmeans-serial: kmeans-serial.c
	gcc kmeans-serial.c -lm -o kmeans-serial

kmeans-pthread: kmeans-pthread.c
	gcc kmeans-pthread.c -pthread -lm -o kmeans-pthread

clean:
	rm kmeans-serial
	rm kmeans-pthread

