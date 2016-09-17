all: kmeans-serial

kmeans-serial: kmeans-serial.c
	gcc kmeans-serial.c -lm -o kmeans-serial

clean:
	rm kmeans-serial

