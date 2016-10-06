all:
	cd serial && $(MAKE)
	cd pthreads && $(MAKE)

serial:
	cd serial && $(MAKE)

pthreads:
	cd pthreads && $(MAKE)

