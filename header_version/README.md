## Generates Sorting Network using SIMD instructions

- Works for 1, 2, 4, and 8 byte types that can be sorted as integers
  (i.e floats and ints)


#### Network Algorithms

- Bitonic
    - Batcher's bitonic algorithm
    - ```vsort::bitonic```
- Batcher
    - Batcher's Merger Exchange algorithm
    - ```vsort::batcher```
- Bosenelson
    - Bose-Nelson algorithm
    - ```vsort::bosenelson```
- Oddeven Merge
    - Batcher's Odd-Even Merge algorithm
    - ```vsort::oddeven```
- Balanced
    - ```vsort::balanced```
- Minimum
    - Uses a variety of algorithms described
      [here](http://users.telenet.be/bertdobbelaere/SorterHunter/sorting_networks.html). Only
      applicable for N = [4, 32]. Otherwise defaults to **Bitonic**.
    - ```vsort::misc```
- Best
    - Uses best algorithm based on my testing. Often this is a power
      of 2 bitonic. For non power of 2 N values will fill the vector
      with max.
    - ```vsort::best```
    
    
#### API Template Signature

- All API functions have the following template
```
template<typename T,
         uint32_t n,
         template<uint32_t _n> typename network = vsort::best,
         simd_instructions simd_set     = vop::simd_instructions_default,
         builtin_usage     builtin_perm = vop::builtin_perm_default>
```
- ```typename T```
    - Type that is being sorted. This is used for determining the
      proper SIMD instructions and proper signedness.
- ```uint32_t n```
    - Number of elements to sort.
- ```template<uint32_t _n> typename network = vsort::best```
    - Network being used. Possible algorithms are described above
- ```simd_instructions simd_set = vop::simd_instructions_default```
    - Flag for which AVX instructions you want to use. If you have to
      restrict the instructions being used beyond what the computer
      has access to (i.e you have ```AVX512``` support but only want
      to use ```AVX2``` you can set
      this. ```vop::simd_instructions_default``` will be set to
      ```vsort::simd_instructions::AVX512``` if you have any
      ```AVX512``` instructions, otherwise it will be set to
      ```vsort::simd_instructions::AVX2```.
- ```builtin_usage builtin_perm = vop::builtin_perm_default```
    - Flag for which permutation implementation you want to use. Both
      GCC and Clang have ```__builtin_shuffle``` and
      ```__builtin_shufflevector``` respectively which do a reasonably
      good job. Otherwise you will live with my own
      optimizations. Generally I would say Clang > mine > GCC. If you
      are compiling without GCC or Clang then you will default to
      mine. The three options here are
      ```vsort::builtin_usage::BUILTIN_FIRST```,
      ```vsort::builtin_usage::BUILTIN_FALLBACK```, or
      ```vsort::builtin_usage::BUILTIN_NONE```.
        - BUILTIN_FIRST
            - will only use GCC/Clang's builtin shuffle
        - BUILTIN_FALLBACK (suggested)
            - will use GCC/Clang's builtin shuffle if its not an
              optimization case I think I did pretty well / better
        - BUILTIN_NONE
            - will only use mine.


#### API Function Names

- ```vsort::sorta(T * const)```
    - Sorts array that is stored in aligned memory and that contains
      at least enough memory to fill an ```xmm```, ```ymm```, or
      ```zmm``` register.
- ```vsort::sortu(T * const)```
    - Sorts array that is either NOT stored in aligned memory or does
      not have enough memory to load into an ```xmm```, ```ymm```, or
      ```zmm``` register.
- ```vsort::sortv(vop::vec_t<T, n>)```
    - Sorts the elements (1, 2, 4, or 8 byte elements) in an
      ```xmm```, ```ymm```, or ```zmm``` register.
      
- NOTE: the register that will be used is determined by sort size *
  size of sort element rounded up to the next power of 2. This means
  that no more than 64 byte can be sorted i.e 64 int8's or 16 int32's
  (32 bytes if you don't have AVX512).
  
  
#### Caveats

- You have must ```AVX2``` or ```AVX512```.
- This has only be tested on Linux ```Ubuntu 20.04```
- This has not been tested as robustly as it probably should have
- For any ```n``` and ```T``` such as ```n * sizeof(T) >
  sizeof(__m256i)``` requires ```AVX512```
- It can take a while to compile for larger ```n``` values with some
  of the networks (hint: dont use ```bosenelson```).


#### Performance Tips

- Dont use ```bosenelson``` or ```balanced```.
- Use ```best``` unless you want to test things
- Unless you have ```AVX512``` instructions don't use
  ```BUILTIN_NONE``` for a non power of 2 N with 1 or 2 byte elements.
      

#### Files

- src/
    - All .h files for creating the networks / the sort instructions
- src/networks
    - Code for creating the networks. These are stored as
      ```std::integer_sequence``` with every even-odd value pair being
      a comparison pair for the sorting network. **misc** has some
      examples of hard coded networks.
- src/instructions
    - Code for the SIMD instructions.
- src/util
    - Some basic util functions used throughout the code base
- src/vec_sort
    - Contains ```vsort::sorta``` and ```vsort::sortu``` API. As well
      the code for creating the compare exchange calls from the
      sorting network
      
- test/
    - Files for testing performance / correctness


#### Examples
```
    #include <vec_sort/vec_sort.h>
    
    uint32_t arr4_not_aligned[4];
    vsort::sortu<uint32_t, 4>(arr4_not_aligned);

    uint32_t arr4_aligned[4] __attribute__((aligned(sizeof(__m256i))));
    vsort::sorta<uint32_t, 4, vsort::bitonic>(arr4_aligned);

    __m256i v = _mm256_loadu_epi32((void * const)arr4_not_aligned);
    v = vsort::sortv<uint32_t, 4, vsort::oddeven, vsort::simd_instructions::AVX2>(v);

    uint8_t arr64_not_aligned[64];
    vsort::sortu<uint8_t, 64, vsort::oddeven, vsort::simd_instructions::AVX512, vsort::builtin_usage::BUILTIN_FALLBACK>(arr64_not_aligned);
```


