#include <stdarg.h>
#include <sys/time.h>

static inline unsigned ticks() {
  struct timeval tv;
  if(gettimeofday(&tv, NULL) != 0)
    return 0;
  return (tv.tv_sec*1000000)+(tv.tv_usec);
}
