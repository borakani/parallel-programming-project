NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++14 -arch=sm_75 -Wno-deprecated-gpu-targets
TARGET    := image_processing_cuda
SRC       := image_processing_cuda.cu
IMAGES    := ../../images

.PHONY: all clean run run-all

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Single run: make run IMG=512x512.pgm BLOCK=16
run: $(TARGET)
	./$(TARGET) $(IMAGES)/$(IMG) $(BLOCK)

# Run all sizes × all block sizes (8, 16, 32)
run-all: $(TARGET)
	@for img in 512x512.pgm 1024x1024.pgm 7680x4320.pgm; do \
	  for bs in 8 16 32; do \
	    echo ">>> $$img | block $$bs"; \
	    ./$(TARGET) $(IMAGES)/$$img $$bs; \
	  done; \
	done

clean:
	rm -f $(TARGET) output_*_blurred.pgm output_*_edges.pgm
