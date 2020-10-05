Creates sorting network using avx
- epi8 / epi64 don't work (need to fix vec_sort_primitives.h blend at the very least)
- if you specify a sorting network that has more bytes than __m512i register its UB

get sorting networking from: https://pages.ripco.net/~jgamble/nw.html

Choose an algorithm i.e Batcher's Merge Exchange and copy the output to a file named:

- [anything without a dash]-[N]

#### do not include a file extension (or make create_network.py more robust)

$> ./create_network <your file>

will output C++ code for the sorting network that uses moderately optimized SIMD instructions (if you really need it optimized a lot of the permutations can be replaced with shuffle and there are probably a ton of nuances I am missing)

i.e

$> ls merge_exhange/
me-16  me-32  me-4  me-8

has Batcher's Merge Exchange sorting networking for N = 4, 8, 16, 32

To create the sort class for N = 8 run:

$> ./create_network merge_exchange/me-8 

to produce the C++ code




