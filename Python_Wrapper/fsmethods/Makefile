CC = g++
PYLIBPATH = $(shell python-config --exec-prefix)/lib
LIB = -L$(PYLIBPATH) $(shell python-config --libs) -lboost_python
OPTS = $(shell python-config --include) -O2

all: RMRMR_rm.so JMI_rm.so MRMR_rm.so VMI_rm.so VMI_gm.so VMI_in.so

JMI_rm.so: JMI_rm.o
	$(CC) $(LIB) -Wl,-rpath,$(PYLIBPATH) -shared  $< -o $@

JMI_rm.o: JMI_rm.cpp
	$(CC) $(OPTS) -fPIC -c $< -o $@

MRMR_rm.so: MRMR_rm.o
	$(CC) $(LIB) -Wl,-rpath,$(PYLIBPATH) -shared  $< -o $@

MRMR_rm.o: MRMR_rm.cpp
	$(CC) $(OPTS) -fPIC -c $< -o $@

RMRMR_rm.so: RMRMR_rm.o
	$(CC) $(LIB) -Wl,-rpath,$(PYLIBPATH) -shared  $< -o $@

RMRMR_rm.o: RMRMR_rm.cpp
	$(CC) $(OPTS) -fPIC -c $< -o $@

VMI_rm.so: VMI_rm.o
	$(CC) $(LIB) -Wl,-rpath,$(PYLIBPATH) -shared  $< -o $@

VMI_rm.o: VMI_rm.cpp
	$(CC) $(OPTS) -fPIC -c $< -o $@
	
VMI_gm.so: VMI_gm.o
	$(CC) $(LIB) -Wl,-rpath,$(PYLIBPATH) -shared  $< -o $@

VMI_gm.o: VMI_gm.cpp
	$(CC) $(OPTS) -fPIC -c $< -o $@

VMI_in.so: VMI_in.o
	$(CC) $(LIB) -Wl,-rpath,$(PYLIBPATH) -shared  $< -o $@

VMI_in.o: VMI_in.cpp
	$(CC) $(OPTS) -fPIC -c $< -o $@

clean:
	rm *.o *.so
