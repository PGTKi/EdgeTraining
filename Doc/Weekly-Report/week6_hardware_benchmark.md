# 硬件 benchmark 记录

> 本文档记录 CPU frequency 和 memory bandwidth 的测试结果。
> 目的是确认硬件本身无故障
> 
> 参考 https://github.com/OAID/Tengine/blob/master/doc/benchmark.md

## 设置 CPU 的频率为最高

```
# cpu{0,1,2,3} 是小核 cpu{4,5} 是大核
# 需要切换到 root
cd /sys/devices/system/cpu/cpu5/cpufreq/
cat scaling_max_freq > scaling_min_freq
```

## 测量 CPU 的频率

[Estimating Arm CPU clock frequencies](http://uob-hpc.github.io/2017/11/22/arm-clock-freq.html)

```
$ taskset -c 5 ./freq
Runtime (seconds)     = 0.555925
Instructions executed = 1000003552
Estimated frequency   = 1798.81 MHz
```

频率大约1.8GHz, 与之前 `procfs` 和商品宣传数据一致

## 测量 memory bandwidth

https://github.com/ssvb/tinymembench

```
$ CFLAGS="-O2 -march=armv8-a -mtune=cortex-a72" make
$ taskset -c 5 ./tinymembench

tinymembench v0.4.9 (simple benchmark for memory throughput and latency)

==========================================================================
== Memory bandwidth tests                                               ==
==                                                                      ==
== Note 1: 1MB = 1000000 bytes                                          ==
== Note 2: Results for 'copy' tests show how many bytes can be          ==
==         copied per second (adding together read and writen           ==
==         bytes would have provided twice higher numbers)              ==
== Note 3: 2-pass copy means that we are using a small temporary buffer ==
==         to first fetch data into it, and only then write it to the   ==
==         destination (source -> L1 cache, L1 cache -> destination)    ==
== Note 4: If sample standard deviation exceeds 0.1%, it is shown in    ==
==         brackets                                                     ==
==========================================================================

 C copy backwards                                     :   2961.5 MB/s (0.2%)
 C copy backwards (32 byte blocks)                    :   2958.6 MB/s
 C copy backwards (64 byte blocks)                    :   2954.3 MB/s
 C copy                                               :   2940.1 MB/s
 C copy prefetched (32 bytes step)                    :   2890.6 MB/s
 C copy prefetched (64 bytes step)                    :   2890.1 MB/s
 C 2-pass copy                                        :   2591.1 MB/s (0.1%)
 C 2-pass copy prefetched (32 bytes step)             :   2641.1 MB/s
 C 2-pass copy prefetched (64 bytes step)             :   2641.9 MB/s
 C fill                                               :   4861.0 MB/s (0.3%)
 C fill (shuffle within 16 byte blocks)               :   4847.9 MB/s
 C fill (shuffle within 32 byte blocks)               :   4855.1 MB/s (0.2%)
 C fill (shuffle within 64 byte blocks)               :   4863.2 MB/s (0.3%)
 ---
 standard memcpy                                      :   2950.4 MB/s
 standard memset                                      :   4857.4 MB/s (0.3%)
 ---
 NEON LDP/STP copy                                    :   2945.8 MB/s (0.1%)
 NEON LDP/STP copy pldl2strm (32 bytes step)          :   2957.1 MB/s
 NEON LDP/STP copy pldl2strm (64 bytes step)          :   2956.3 MB/s
 NEON LDP/STP copy pldl1keep (32 bytes step)          :   2874.1 MB/s
 NEON LDP/STP copy pldl1keep (64 bytes step)          :   2874.9 MB/s
 NEON LD1/ST1 copy                                    :   2944.4 MB/s
 NEON STP fill                                        :   4857.8 MB/s (0.3%)
 NEON STNP fill                                       :   4784.3 MB/s
 ARM LDP/STP copy                                     :   2942.0 MB/s
 ARM STP fill                                         :   4855.3 MB/s (0.4%)
 ARM STNP fill                                        :   4797.9 MB/s (0.3%)

==========================================================================
== Memory latency test                                                  ==
==                                                                      ==
== Average time is measured for random memory accesses in the buffers   ==
== of different sizes. The larger is the buffer, the more significant   ==
== are relative contributions of TLB, L1/L2 cache misses and SDRAM      ==
== accesses. For extremely large buffer sizes we are expecting to see   ==
== page table walk with several requests to SDRAM for almost every      ==
== memory access (though 64MiB is not nearly large enough to experience ==
== this effect to its fullest).                                         ==
==                                                                      ==
== Note 1: All the numbers are representing extra time, which needs to  ==
==         be added to L1 cache latency. The cycle timings for L1 cache ==
==         latency can be usually found in the processor documentation. ==
== Note 2: Dual random read means that we are simultaneously performing ==
==         two independent memory accesses at a time. In the case if    ==
==         the memory subsystem can't handle multiple outstanding       ==
==         requests, dual random read has the same timings as two       ==
==         single reads performed one after another.                    ==
==========================================================================

block size : single random read / dual random read
      1024 :    0.0 ns          /     0.0 ns 
      2048 :    0.0 ns          /     0.0 ns 
      4096 :    0.0 ns          /     0.0 ns 
      8192 :    0.0 ns          /     0.0 ns 
     16384 :    0.0 ns          /     0.0 ns 
     32768 :    0.0 ns          /     0.0 ns 
     65536 :    4.5 ns          /     7.2 ns 
    131072 :    6.8 ns          /     9.7 ns 
    262144 :    9.9 ns          /    12.8 ns 
    524288 :   11.4 ns          /    14.7 ns 
   1048576 :   16.8 ns          /    23.5 ns 
   2097152 :  106.6 ns          /   161.7 ns 
   4194304 :  148.9 ns          /   201.9 ns 
   8388608 :  174.9 ns          /   222.9 ns 
  16777216 :  187.9 ns          /   231.5 ns 
  33554432 :  195.5 ns          /   237.0 ns 
  67108864 :  207.8 ns          /   254.7 ns 
```

这个大约 4GB/s 的数据也符合预期
