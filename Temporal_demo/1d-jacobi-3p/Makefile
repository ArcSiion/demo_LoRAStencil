EXE := exe-1d-jacobi-3p
CC ?= gcc

CFLAGS ?= -O3 -march=native -mtune=native -mavx512f -mfma -fopenmp -Wall -Wextra
CPPFLAGS ?= -DCHECK
LDLIBS ?= -lm

SRCS := main.c naive_scalar.c vectime_stride7_512.c

.PHONY: all clean run

all: $(EXE)

$(EXE): $(SRCS) defines.h defines_512.h ../common.h
	$(CC) $(CPPFLAGS) $(CFLAGS) $(SRCS) -o $@ $(LDLIBS)

run: $(EXE)
	./$(EXE) 1024 16

clean:
	rm -f $(EXE)
