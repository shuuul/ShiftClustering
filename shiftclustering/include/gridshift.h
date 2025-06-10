#include <map>
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;


void grid_cluster(int n,
                  int d,
                  int base,
                  int iterations,
                  float bandwidth,
                  int * offsets,
                  float * X_shifted,
                  int * membership,
                  int * k_num) {

    map< vector<int>, pair< vector<float>, int> > cluster_grid;
    map< vector<int>, int > map_cluster;
    map< int, int > clus;
    map< int, int > :: iterator it2;
    map< vector<int>, pair< vector<float>, int> >:: iterator it;
    map< vector<int>, pair< vector<float>, int> > means;


    int iter = 0;
    vector<int> current_bin(d);
    vector<int> bin(d);
    vector<int> membershipp(n);
    vector<int> membershipp_old(n);

    // new clustering at grids - initial binning
    int temp = 0;

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < d; k++) {
            // On dimension k, assign the bin to point i
            bin[k] = X_shifted[i * d + k] / bandwidth;
        }
        // If the current_bin is not already in the 'means' map, it is initialized
        if (cluster_grid.find(bin) == cluster_grid.end()) {
            cluster_grid[bin] = make_pair(std::vector<float>(d, 0), 0);
        }
        // Add the current point to the sum vector
        for (int k = 0; k < d; k++){
            cluster_grid[bin].first[k] += X_shifted[i * d + k];
        }
        // Increment the count of bin
        cluster_grid[bin].second++;
        // Update active grid cells
        if (map_cluster.find(bin) == map_cluster.end()){
            map_cluster[bin] = temp++;
        }
        // Update the inverse map
        membershipp[i] = map_cluster[bin] * 1.0;
    }

    while (iter <= iterations){
        iter++;
        means.clear();
        
        // Process cluster grid updates
        for (it = cluster_grid.begin(); it != cluster_grid.end(); ++it ){
            for (int j = 0; j < pow(base, d); j++) {
                for (int k = 0; k < d; k++) {
                    current_bin[k] = it->first[k] + offsets[j * d + k];
                    if (j == 0){
                        bin[k] =  it->first[k] ;
                    }
                }

                // If neighbor exists, add it to the mean
                if (cluster_grid.find(current_bin) != cluster_grid.end()) {
                    if (means.find(current_bin) == means.end()) {
                        means[current_bin] = make_pair(std::vector<float>(d, 0), 0);
                    }

                    for (int k = 0; k < d; k++) {
                        means[current_bin].first[k] += cluster_grid[bin].first[k] * 1.0;
                    }
                    means[current_bin].second += cluster_grid[bin].second;
                }
            }
        }

         for (it = cluster_grid.begin(); it != cluster_grid.end(); ++it ){
            for (int k = 0; k < d; k++) {
                current_bin[k] = it->first[k];
            }

            for (int k = 0; k < d; k++) {
                cluster_grid[current_bin].first[k] = means[current_bin].first[k] * 1.0 / means[current_bin].second;
            }
        }

        // update cluster grid and membership
        map< vector<int>, pair< vector<float>, int> > cluster_grid_old = cluster_grid;
        map< vector<int>, int > map_cluster_old = map_cluster;

        cluster_grid.clear();
        map_cluster.clear();
        clus.clear();


        int temp = 0;
        for (it = cluster_grid_old.begin(); it != cluster_grid_old.end(); ++it ){

            for (int k = 0; k < d; k++) {
                bin[k] = it->second.first[k] / bandwidth;
                current_bin[k] = it->first[k];
            }

            if (cluster_grid.find(bin) == cluster_grid.end()) {
                cluster_grid[bin] = make_pair(std::vector<float>(d, 0), 0);
            }

            for (int k = 0; k < d; k++){
                cluster_grid[bin].first[k] += it->second.first[k] * 1.0 * it->second.second;
            }
            cluster_grid[bin].second += it->second.second;

            if (map_cluster.find(bin) == map_cluster.end()){
                map_cluster[bin] = temp++;
            }
            clus[map_cluster_old[current_bin]] = map_cluster[bin];
        }

        int break_points = 0;
        for (it2 = clus.begin(); it2 != clus.end(); ++it2) {
            if (it2->first !=  it2->second){
                replace (membershipp.begin(), membershipp.end(), it2->first, it2->second);
                break_points += it2->first - it2->second;
            }
        }
        if (break_points == 0){
            break;
        }
    }
    copy(membershipp.begin(), membershipp.end(), membership);

    vector<int> k_num2(1);
    k_num2[0] = cluster_grid.size();
    vector<float> bins(k_num2[0] * d);
    int itt = 0;
     for (it = cluster_grid.begin(); it != cluster_grid.end(); ++it ){

            for (int k = 0; k < d; k++) {
                bins[itt * d + k] = it->second.first[k] *1.0 / it->second.second;
            }
            itt++;
        }
    copy(bins.begin(), bins.end(), X_shifted);
    copy(k_num2.begin(), k_num2.end(), k_num);
}