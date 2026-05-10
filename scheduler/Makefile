# POSIX (Linux / WSL / macOS Darwin fork destekleri). make clean && make all
CC = cc
CFLAGS = -Wall -Wextra -O2 -std=c11 -D_DEFAULT_SOURCE -D_GNU_SOURCE
LDFLAGS =
RM = rm -f

PROGS = compare_pgms decide_scheduler run_benchmark

RUNOBJS = run_benchmark.o sche_compat.o sche_metric_parse.o \
	sche_metric_avg.o sche_csv.o sche_execposix.o sche_fsutil.o sche_pgm.o

.PHONY: all clean

all: $(PROGS)

compare_pgms: compare_pgms.o sche_pgm.o

	$(CC) $(LDFLAGS) -o $@ $^

decide_scheduler: decide_scheduler.o sche_policy.o

	$(CC) $(LDFLAGS) -o $@ $^

run_benchmark: $(RUNOBJS)

	$(CC) $(LDFLAGS) -o $@ $^

%.o: %.c

	$(CC) $(CFLAGS) -c $< -o $@

clean:

	$(RM) *.o $(PROGS)
