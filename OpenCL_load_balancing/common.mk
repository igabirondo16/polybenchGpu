INCLUDE := -I/usr/include
LIBPATH := -L/lib/x86_64-linux-gnu
LIB     := -pthread -lOpenCL -lm

# You must define these
#CFILES := main.c            # Change this to your source files
#EXECUTABLE := my_opencl_app # Change this to your desired output name

# Build rule
all:
	@echo "--- Compiling $(EXECUTABLE) ---"
	@echo "CFILES = $(CFILES)"
	@echo "INCLUDE = $(INCLUDE)"
	@echo "LIBPATH = $(LIBPATH)"
	@echo "LIB = $(LIB)"
	@echo "Full command: gcc -O3 $(INCLUDE) $(LIBPATH) $(CFILES) -o $(EXECUTABLE) $(LIB)"
	@gcc -O3 $(INCLUDE) $(LIBPATH) $(CFILES) -o $(EXECUTABLE) $(LIB)

# Clean rule
clean:
	@rm -f *~ *.exe *.o $(EXECUTABLE)
