CXX=g++

INCLUDES=-I.
CXXFLAGS=-O3 -std=c++17 -march=native -mtune=native -mavx2


all:
	$(CXX) $(CXXFLAGS) $(INCLUDES) timing/stats.cc -c
	$(CXX) $(CXXFLAGS) $(INCLUDES) -DTEST_TYPE=$(TEST_TYPE) -DTEST_N=$(TEST_N) -DTEST_SIMD=$(TEST_SIMD) -DTEST_BUILTIN=$(TEST_BUILTIN) driver.cc -c
	$(CXX) $(CXXFLAGS) $(INCLUDES) driver.o stats.o -o driver

clean:
	rm -f *~ *#* *.o driver
