Create Sorting Network for N in [2, 32]

Uses: https://github.com/jgamble/Algorithm-Networksort for creating
the networks


All algorithms supported in Algorithm-Networksort are supported by
this.



## Files

#### create_sorters.py
- helper script, this will create the implementation of the sorting
  network and store it in file specified by '--dest' arguments. As
  well it will automatically include the implementation file in the
  file specified by the '--header_file' argument. See '-h' for more
  usage details.

#### driver.cc
- simple test program with ```main()``` for correctness / performance
  test. Current compares SIMD sorting network to ```std::sort```.

#### generation/create_network_impl.py
- script that actually creates the SIMD sorting networking
  implementation

#### generation/create_all_networks.py
- script for creating the network specifier file for a given algorithm
  and N value (or for all algorithms and all N [2, 32]). This is just
  a wrapper for ```create_networks.pl```.

#### generation/create_networks.pl
- script that invokes https://github.com/jgamble/Algorithm-Networksort
  API to create the sorting network specifier file.

#### Makefile
- Shitty Makefile

#### README.md
- This

#### implementation/
- Default directory for where ```create_sorters.py``` will store
  implementation files of SIMD sorting network

#### networks/
- Directory containing all sorting network specifier files. This is
  just the output of ```create_networks.pl``` for N [2, 32] and all
  available algorithms. The structure is ```networks/<algorithm
  name>/<algorithm name>-<N>```. It is important that the files
  (irrelivant of directory) are stored in the former ```<algorithm
  name>-<N>``` as the python scripts rely on this.

#### util/cpp_attributes.h
- Just some ```#define``` macros I use a lot

#### util/constexpr_util.h
- Right now just ```ulog2```

#### sort_base/vec_sort.h
- Default location for storing ```#include``` of SIMD sorting
  implementations generated by ```create_sorters.py```. This is also
  included by ```driver.cc```.

#### sort_base/vec_sort_incl.h
- Header file included in all generated SIMD sorting networks. This
  has all the other necessary includes as well as the struct
  definition that each sort function is a specilization of.

#### sort_base/vec_sort_primitives.h
- Templated wrappers for SIMD functions used by the SIMD sorting
  networks as well as a mildly optimized permutation function and the
  ```compare_exchange``` function used by all the SIMD networks.


## Shortcomings

#### epi8 and epi64 have some bugs
#### right now avx512 is required for many of the algorithms
#### non power of 2 values won't work unless you specify an array with enough memory and manually set indexes outside of N to max

