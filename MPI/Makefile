MCC=mpicc
CC=gcc
CFLAGS=-O3 
LIBFLAGS=-lm

OBJDIR = obj

_OBJ = args.o data.o setup.o vtk.o maxwell.o
OBJ = $(patsubst %,$(OBJDIR)/%,$(_OBJ))

.PHONY: directories

all: directories maxwell

obj/%.o: %.c
	$(MCC) -c -o $@ $< $(CFLAGS) 

maxwell: $(OBJ)
	$(MCC) -o $@ $^ $(CFLAGS) $(LIBFLAGS) 

clean:
	rm -Rf $(OBJDIR)
	rm -f maxwell

directories: $(OBJDIR)

$(OBJDIR):
	mkdir -p $(OBJDIR)

