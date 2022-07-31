CC=gcc
CFLAGS=-O3 
LIBFLAGS=-lm

OBJDIR = obj

_OBJ = args.o data.o setup.o vtk.o maxwell.o
OBJ = $(patsubst %,$(OBJDIR)/%,$(_OBJ))

.PHONY: directories

all: directories maxwell

obj/%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS) 

maxwell: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LIBFLAGS) 

clean:
	rm -Rf $(OBJDIR)
	rm -f maxwell

directories: $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

