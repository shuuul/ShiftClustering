#include <map>
#include <vector>
#include <queue>
#include <set>
#include <algorithm>

using namespace std;


void shift_cy(int n,
              int d,
              int base,
              float bandwidth,
              int * offsets,
              float * X_shifted) {
    /*
        Generate 3**d neighbors for any point.

        Parameters
        ----------
        n: Number of samples
        d: Dimension
        base: 3, corresponding to (-1, 0, 1)
        bandwith: Radius for binning points. Points are assigned to the
                  (d, ) bin corresponding to floor division by bandwidth
        offsets: (3**d, d) array of offsets to be added to
                 a bin to get its neighbors
        X_shifted: (n, d) array of new points after one iteration
                   of shift

    */

    // bin -> (sum, count)
    map< vector<int>, pair< vector<float>, int> > means;

    // For each point, the bin position in each dimension
    int * bins = new int[n * d];

    // Store the current bin during the iterations
    vector<int> current_bin(d);

    // First pass: bin all points
    for (int i = 0; i < n; i++) {
        // Bin point
        for (int k = 0; k < d; k++) {
            // On dimension k, assign the bin to point i
            bins[i * d + k] = X_shifted[i * d + k] / bandwidth;
            current_bin[k] = bins[i * d + k];
        }

        // If the current_bin is not already in the 'means' map, it is initialized
        if (means.find(current_bin) == means.end()) {
            means[current_bin] = make_pair(std::vector<float>(d, 0), 0);
        }
    }

    // Second pass: accumulate means
    for (int i = 0; i < n; i++) {
        // scan the bins in each dimension, 3*3*3 positions
        for (int j = 0; j < pow(base, d); j++) {
            // Get neighbor
            for (int k = 0; k < d; k++) {
                 current_bin[k] = bins[i * d + k] + offsets[j * d + k];
            }
            // If the neighbor bin exists in the means map,
            // the point's coordinates are added to the sum vector,
            // and the count is incremented.
            if (means.find(current_bin) != means.end()) {
                for (int k = 0; k < d; k++) {
                    means[current_bin].first[k] += X_shifted[i * d + k];
                }
                means[current_bin].second++;
            }
        }
    }

    // Third pass: set every point to the mean of its neighbors
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < d; k++) {
            current_bin[k] = bins[i * d + k];
        }

        for (int k = 0; k < d; k++) {
            X_shifted[i * d + k] = means[current_bin].first[k] * 1.0 / means[current_bin].second;
        }
    }
    delete[] bins;
}