all:
	make serial
	make pthread

serial: kmeans-serial.c
	gcc kmeans-serial.c -g -lm -o kmeans-serial

pthread: kmeans-pthread.c info.h
	gcc kmeans-pthread.c -g -pthread -lm -o kmeans-pthread

clean:
	rm kmeans-serial
	rm kmeans-pthread
