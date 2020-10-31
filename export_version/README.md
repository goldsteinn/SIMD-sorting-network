## export_version

#### Algorithms:
- bitonic
- oddeven
- batcher
- minimum
- bosenelson
- best
    - best involves a fair amount of computation. It will select on of the above. (generally bitonic or minimum). 

#### Types:
- ```int8_t``` and ```uint8_t```
- ```int16_t``` and ```uint16_t```
- ```int32_t``` and ```uint32_t```
- ```int64_t``` and ```uint64_t```

Note if you want to ```float``` or ```double``` just use ```int32_t``` or ```int64_t```

#### N:
- Maximum N is determined by the sizeof the largest SIMD register your machine support. The exact algorithm is: ```N <= sizeof(__SIMD_reg) / sizeof(Sort_Type)```


#### Extra Flags:
- **--optimization** = ["space", "uop"]
    - instruction selection will break ties by minimizing .rodata or uops respectively
- **--int-aligned**
    - set if you can gurantee any array being sorted will have accessible memory up to: ```arr + Roundup_To_Sizeof_Int(sizeof(Sort_Type) * N)```
- **--extra-memory**
    - set if you can gurantee sizeof SIMD register is available starting from ```arr```
- **--aligned**
    - set if you can gurantee any array being sorted is aligned to sizeof SIMD register
- **--constraint**
    - set to provide additional instruction set constraints beyond what the machine this script is being run on supports. I.e ```--constraint AVX512``` would prohibit all ```AVX512``` instructions irrelivant of what the machine supports. Note ```AVX2``` is required.
    
    
#### More Info
- ```./export.py -h```

#### Output
- ```__SIMD_register <algorithm_name>_<sort_size>_<sort_type>_vec(__SIMD_register)```
- ```void <algorithm_name>_<sort_size>_<sort_type>(sort_type * arr)```


