CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -pedantic
LDFLAGS = -lm

TARGET = models/run_models
SOURCES = models/main.c utility/utils.c

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $@ $(SOURCES) $(LDFLAGS)

run: $(TARGET)
	cd models && ./run_models

clean:
	rm -f $(TARGET)

.PHONY: all run clean
