#ifndef MSZ_CPU_INTERNAL_H
#define MSZ_CPU_INTERNAL_H

#include <vector>
#include <cmath>
#include <atomic>
#include "api/MSz.h"

void computeAdjacency_cpu(std::vector<int>& adjacency, int width, int height, int depth, int maxNeighbors);

int find_direction_cpu(const std::vector<double>* data, const std::vector<int>* adjacency, std::vector<int>& direction_as, 
                std::vector<int>& direction_ds, int width, int height, 
                int depth, int maxNeighbors);

void initializeWithIndex_cpu(std::vector<int>& label, const std::vector<int>* direction_ds, const std::vector<int>* direction_as);

void mappath_cpu(std::vector<int>& label, const std::vector<int> *direction_as, const std::vector<int> *direction_ds, int width, int height, int depth, int maxNeighbors);

void get_false_criticle_points_cpu(int &count_f_max, int &count_f_min, const std::vector<int> *adjacency,
                                const std::vector<double>* decp_data, const std::vector<int>* or_direction_as,
                                const std::vector<int>* or_direction_ds, std::vector<int> &false_min, std::vector<int> &false_max, int maxNeighbors, int data_size);

void initialization_cpu(std::vector<double> &d_deltaBuffer, int data_size, double bound);

void applyDeltaBuffer_cpu(const std::vector<double> *d_deltaBuffer, const std::vector<double> *input_data, std::vector<double> &decp_data,
                        int data_size, double bound);
int fix_process_cpu(std::vector<int> *or_direction_as,std::vector<int> *or_direction_ds,
        std::vector<int> *de_direction_as,std::vector<int> *de_direction_ds,
        const std::vector<double> *input_data, std::vector<double> *decp_data,
        std::vector<int>* dec_label,std::vector<int>* or_label, 
        int width, int height, int depth, 
        double bound, 
        int preserve_min, int preserve_max, 
        int preserve_path, int neighbor_number);

int count_false_cases_cpu(std::vector<int> *or_direction_as,std::vector<int> *or_direction_ds,
        std::vector<int> *de_direction_as,std::vector<int> *de_direction_ds,
        const std::vector<double> *input_data, std::vector<double> *decp_data,
        std::vector<int>* dec_label,std::vector<int>* or_label, 
        int width, int height, int depth, 
        int neighbor_number,
        int &wrong_min, int &wrong_max, int &wrong_labels);

int extract_critical_points_cpu(
        const std::vector<double> *data,
        std::vector<MSz_critical_point_t> &critical_points,
        unsigned int critical_point_types,
        int width, int height, int depth,
        int neighbor_number);

#endif // MSZ_CPU_INTERNAL_H
