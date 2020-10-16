#ifndef _STATS_H_
#define _STATS_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <vector>

#include <timing/timers.h>


namespace stats {

static constexpr bool human_readable = true;
static constexpr bool csv            = false;

struct stats_out {

    uint64_t N;
    double   median;
    double   mean;
    double   max;
    double   min;

    double p99;
    double p95;
    double p90;

    double stddev;
    double variance;

    timers::time_units units;

    void print(bool format, FILE * outfile = stderr);

    void print_csv(FILE * outfile = stderr);
    void print_hr(FILE * outfile = stderr);

    void get_stats(uint64_t *         data,
                   uint32_t           n,
                   timers::time_units _units = timers::time_units::NSEC);
    void get_stats(std::vector<uint64_t> & data,
                   timers::time_units      _units = timers::time_units::NSEC);

    void get_stats(double *           data,
                   uint32_t           n,
                   timers::time_units _units = timers::time_units::NSEC);
    void get_stats(std::vector<double> & data,
                   timers::time_units    _units = timers::time_units::NSEC);

    void sorted_array_to_stats(
        double *           data,
        uint32_t           n,
        timers::time_units _units = timers::time_units::NSEC);
};

}  // namespace stats

#endif
