#include <timing/stats.h>

#include <util/verbosity.h>
#include <algorithm>
#include <numeric>

namespace stats {
static constexpr uint64_t sci_notation_bound = 10 * 1000;

static const char possible_fmts[2][8] = { "%.4E", "%.4lf" };
static const char *
best_format(double value) {
    if (value >= (sci_notation_bound)) {
        return possible_fmts[0];
    }
    return possible_fmts[1];
}

static void
stddev_and_variance(double   mean,
                    double * data,
                    uint32_t n,
                    double * stddev_out,
                    double * variance_out) {
    if (n <= 1 || data == NULL) {
        errv_print(
            "Error: Insufficient data for computing stddev or variance\n");
        return;
    }

    double sum = 0;
    for (uint32_t i = 0; i < n; ++i) {
        sum += (data[i] - mean) * (data[i] - mean);
    }
    if (stddev_out != NULL) {
        *stddev_out = sum / ((double)(n - 1));
    }
    if (variance_out != NULL) {
        *variance_out = sum / ((double)(n));
    }
}

void
stats_out::print(bool format, FILE * outfile) {
    if (format == human_readable) {
        print_hr(outfile);
    }
    else {
        print_csv(outfile);
    }
}

void
stats_out::print_hr(FILE * outfile) {
    char fmt_buf[512] = "";
    sprintf(fmt_buf,
            "----------------------------------------\n"
            "N          : %lu\n"
            "Median     : %s %s\n"
            "Mean       : %s %s\n"
            "Max        : %s %s\n"
            "Min        : %s %s\n"
            "P99        : %s %s\n"
            "P95        : %s %s\n"
            "P90        : %s %s\n"
            "stddev     : %s %s\n"
            "variance   : %s %s\n"
            "----------------------------------------\n",
            N,
            best_format(median),
            timers::unit_to_str(units),
            best_format(mean),
            timers::unit_to_str(units),
            best_format(max),
            timers::unit_to_str(units),
            best_format(min),
            timers::unit_to_str(units),
            best_format(p99),
            timers::unit_to_str(units),
            best_format(p95),
            timers::unit_to_str(units),
            best_format(p90),
            timers::unit_to_str(units),
            best_format(stddev),
            timers::unit_to_str(units),
            best_format(variance),
            timers::unit_to_str(units));

    fprintf(outfile,
            fmt_buf,
            median,
            mean,
            max,
            min,
            p99,
            p95,
            p90,
            stddev,
            variance);
}

void
stats_out::print_csv(FILE * outfile) {
    fprintf(outfile,
            "n,median,mean,max,min,p99,p95,p90,stddev,variance\n"
            "%lu,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf\n",
            N,
            median,
            mean,
            max,
            min,
            p99,
            p95,
            p90,
            stddev,
            variance);
}

void
stats_out::sorted_array_to_stats(double *           data,
                                 uint32_t           n,
                                 timers::time_units _units) {

    units = _units;
    if (n == 1) {
        N        = 1;
        median   = data[0];
        mean     = data[0];
        max      = data[0];
        min      = data[0];
        p99      = data[0];
        p95      = data[0];
        p90      = data[0];
        stddev   = 0;
        variance = 0;
    }
    else {
        N = n;
        median =
            (n % 2) ? (data[n / 2]) : ((data[n / 2] + data[(n / 2) + 1]) / 2);
        mean = std::accumulate(data, data + n, 0) / n;
        max  = data[n - 1];
        min  = data[0];

        uint64_t p99_idx = .99 * ((double)n);
        uint64_t p95_idx = .95 * ((double)n);
        uint64_t p90_idx = .90 * ((double)n);
        p99              = data[p99_idx];
        p95              = data[p95_idx];
        p90              = data[p90_idx];
        stddev_and_variance(mean, data, n, &stddev, &variance);
    }
}

void
stats_out::get_stats(uint64_t * data, uint32_t n, timers::time_units _units) {
    double * data_dbl = (double *)data;
    for (uint32_t i = 0; i < n; ++i) {
        double temp = data[i];
        data_dbl[i] = temp;
    }
    get_stats(data_dbl, n, _units);
}

void
stats_out::get_stats(std::vector<uint64_t> & data, timers::time_units _units) {
    get_stats(data.data(), data.size(), _units);
}


void
stats_out::get_stats(double * data, uint32_t n, timers::time_units _units) {
    if (n == 0 || data == NULL) {
        errv_print("Error: No data to collect stats on\n");
        return;
    }
    std::sort(data, data + n);
    sorted_array_to_stats(data, n, _units);
}

void
stats_out::get_stats(std::vector<double> & data, timers::time_units _units) {
    get_stats(data.data(), data.size(), _units);
}


}  // namespace stats
