#ifndef MSZ_GLOBAL_INTERNAL_H
#define MSZ_GLOBAL_INTERNAL_H

#include <vector>
#include <cmath>
#include <atomic>

extern "C" {
    extern int directions[78], direction_to_index_mapping[26][3]; 
    int getDirection(int x, int y, int z, int maxNeighbors);

    int from_direction_to_index(int cur, int direc, int width, int height, int depth, int maxNeighbors);

    void fix_maxi_critical(const std::vector<double> *input_data, const std::vector<double> *decp_data,
                            const std::vector<int> *or_direction_as, const std::vector<int> *or_direction_ds, 
                            const std::vector<int> *de_direction_as, const std::vector<int> *de_direction_ds, 
                            std::vector<double> &d_deltaBuffer,
                            int width, int height, int depth, int maxNeighbors, double bound,
                            int index, int direction);
    
    double atomicCASDouble(double* ptr, double old_val, double new_val);

    void swap(int index, double delta, std::vector<double> &d_deltaBuffer);

    int fixpath(const std::vector<double> *input_data, const std::vector<double> *decp_data,
                        const std::vector<int> *or_direction_as, const std::vector<int> *or_direction_ds, 
                        const std::vector<int> *de_direction_as, const std::vector<int> *de_direction_ds, 
                        std::vector<double> &d_deltaBuffer,int index, int direction, std::atomic<int>* id_array, double bound, int width, int height, int depth, int maxNeighbors);
    
    double get_wrong_index_path(const std::vector<int> *or_label, const std::vector<int> *dec_label, std::vector<int> &wrong_index_as, std::vector<int> &wrong_index_ds, int data_size);
    double count_false_labels(const std::vector<int> *or_label, const std::vector<int> *dec_label, int data_size);
}
#endif // MSZ_GLOBAL_INTERNAL_H
