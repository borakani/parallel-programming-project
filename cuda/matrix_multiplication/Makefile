NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++14 -arch=sm_75 -Wno-deprecated-gpu-targets
TARGET    := matrix_mult_cuda
SRC       := matrix_mult_cuda.cu

.PHONY: all clean run run-all

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Single run: make run N=1024 BLOCK=16
run: $(TARGET)
	./$(TARGET) $(N) $(BLOCK)

# Sweep over common sizes and all block sizes (8, 16, 32)
run-all: $(TARGET)
	@for n in 256 512 1024 2048; do \
	  for bs in 8 16 32; do \
	    echo ">>> N=$$n | block $$bs"; \
	    ./$(TARGET) $$n $$bs; \
	  done; \
	done

clean:
	rm -f $(TARGET)
