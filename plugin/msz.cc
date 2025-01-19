#include "api/MSz.h" 
#include "pressio_data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"

namespace libpressio {
namespace msz_false_label_ratio {

class MSzFalseLabelRatioPlugin : public libpressio_metrics_plugin {
public:
    
    int begin_compress_impl(const struct pressio_data*, const struct pressio_data*) override {
        
        return 0;
    }

    
    int end_decompress_impl(const struct pressio_data* input, const struct pressio_data* output, int) override {
        if (!input || !output) {
            return set_error(1, "Input or output data is missing");
        }

        
        const int W = input->dimensions().size() > 0 ? input->dimensions()[0] : 1;
        const int H = input->dimensions().size() > 1 ? input->dimensions()[1] : 1;
        const int D = input->dimensions().size() > 2 ? input->dimensions()[2] : 1;

        const double* original_data = static_cast<const double*>(input->data());
        const double* decompressed_data = static_cast<const double*>(output->data());

        for (int i = 0; i < W * H * D; i++) {
            printf("input[%d]: %f, output[%d]: %f\n", i, original_data[i], i, decompressed_data[i]);
        }
        int num_false_min = 0;
        int num_false_max = 0;
        int num_false_labels = 0;
        
        int status = MSz_count_faults(
            original_data, decompressed_data,
            num_false_min, num_false_max, num_false_labels,
            0,  // connectivity_type: piecewise linear
            W, H, D,
            MSZ_ACCELERATOR_NONE
        );

        if (status == 0) {
            false_label_ratio = static_cast<double>(num_false_labels) / (W * H * D);
        } else {
            set_error(status, "MSz_count_faults failed");
            false_label_ratio.reset();
        }

        return status;


    }

    
    struct pressio_options get_configuration_impl() const override {
        pressio_options opts;
        set(opts, "pressio:stability", "stable");
        set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
        return opts;
    }

    
    struct pressio_options get_documentation_impl() const override {
        pressio_options opts;
        set(opts, "pressio:description", "Computes the ratio of false labeled points in decompressed data");
        set(opts, "msz_false_label_ratio:false_label_ratio", "The ratio of false labeled points to total points");
        return opts;
    }

    
    pressio_options get_metrics_results(pressio_options const&) override {
        pressio_options opts;
        if (false_label_ratio) {
            set(opts, "msz_false_label_ratio:false_label_ratio", *false_label_ratio);
        } else {
            set_type(opts, "msz_false_label_ratio:false_label_ratio", pressio_option_double_type);
        }
        return opts;
    }

    
    std::unique_ptr<libpressio_metrics_plugin> clone() override {
        return compat::make_unique<MSzFalseLabelRatioPlugin>(*this);
    }

    
    const char* prefix() const override {
        return "msz_false_label_ratio";
    }

private:
    compat::optional<double> false_label_ratio;
};


static pressio_register msz_false_label_ratio_plugin(metrics_plugins(), "msz_false_label_ratio", []() {
    return compat::make_unique<MSzFalseLabelRatioPlugin>();
});

}  // namespace msz_false_label_ratio
}  // namespace libpressio
