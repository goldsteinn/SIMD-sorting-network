CXX=g++

INCLUDES=-I.
CXXFLAGS=-O3 -std=c++17 -march=native -mtune=native -mavx2


all:
	$(CXX) $(CXXFLAGS) $(INCLUDES) timing/stats.cc -c
	$(CXX) $(CXXFLAGS) $(INCLUDES) driver.cc -c
	$(CXX) $(CXXFLAGS) $(INCLUDES) driver.o stats.o -o driver

clean:
	rm -f *~ *#* *.o driver
