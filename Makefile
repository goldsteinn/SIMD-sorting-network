

INCLUDES=-I.
CXXFLAGS=-O3 -std=c++17 -march=native -mtune=native -mavx2


all:
	$(CXX) $(CXXFLAGS) $(INCLUDES) driver.cc -o driver

clean:
	rm -f *~ *#* driver
