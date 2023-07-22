#include <map>
#include <vector>
#include <queue>
#include <set>
#include <algorithm>

using namespace std;

void track_cy(int l,
			  int w,
			  int length,
			  int width,
			  float bandwidth,
			  float * X,
			  float * hist,
			  int * mask) {

}


void stitch_cy(int n,
               int m,
               int threshold,
               int n_clusters,
               int * clusters,
               int * updated_clusters) {
    /*
        Stitch together clusters split vertically and horizonally.
        Whenever we encounter a vertical or horizontal edge between
        two clusters with greater than `threshold` pixels, note the
        two clusters as equivalent. In the end, all connected
        components of clusters will be labeled as the same cluster.

        Parameters
        ----------
        clusters: (n, d) array of clusters for a

    */
    int cnt = 1;
    int prev_cluster1, prev_cluster2, cluster1, cluster2;
    map<int, set<int> > equivalent_clusters;

    for (int i = 0; i < n_clusters; i++) {
    	equivalent_clusters[i].insert(i);
    }

    // Stitch horizontally
	for (int i = 0; i < n - 1; i++) {
		prev_cluster1 = clusters[i * m];
		prev_cluster2 = clusters[(i + 1) * m];

		for (int j = 1; j < m; j++) {
			cluster1 = clusters[i * m + j];
			cluster2 = clusters[(i + 1) * m + j];

			if (cluster1 == prev_cluster1 && cluster2 == prev_cluster2 && cluster1 != cluster2) {
				cnt++;
			} else {
				cnt = 1;
			}

			if (cnt > threshold) {
				equivalent_clusters[cluster1].insert(cluster2);
				equivalent_clusters[cluster2].insert(cluster1);
			}

			prev_cluster1 = cluster1;
			prev_cluster2 = cluster2;
		}
	}

	// Stitch vertically
	for (int j = 0; j < m - 1; j++) {
		prev_cluster1 = clusters[j];
		prev_cluster2 = clusters[j + 1];

		for (int i = 1; i < n; i++) {
			cluster1 = clusters[i * m + j];
			cluster2 = clusters[i * m + j + 1];

			if (cluster1 == prev_cluster1 && cluster2 == prev_cluster2 && cluster1 != cluster2) {
				cnt++;
			} else {
				cnt = 1;
			}

			if (cnt > threshold) {
				equivalent_clusters[cluster1].insert(cluster2);
				equivalent_clusters[cluster2].insert(cluster1);
			}

			prev_cluster1 = cluster1;
			prev_cluster2 = cluster2;
		}
	}

	queue<int> q;
	map<int, int> cluster_map;
    int point;
    cnt = 0;

    for (auto c : equivalent_clusters) {

        q = queue<int>();

        if (cluster_map.count(c.first) == 0) {

            q.push(c.first);
            cluster_map[c.first] = cnt;

            while (!q.empty()) {

                point = q.front();
                q.pop();

        		for (auto neighbor : equivalent_clusters[point]) {
                    if (cluster_map.count(neighbor) == 0) {
                        q.push(neighbor);
                        cluster_map[neighbor] = cnt;
                    }

                }

            }

            cnt ++;
        }
    }

    for (int i = 0; i < n * m; i++) {
    	updated_clusters[i] = cluster_map[clusters[i]];
    }
}


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

    map< vector<int>, pair< vector<float>, int> > means;

    int * bins = new int[n * d];
    vector<int> current_bin(d);

    for (int i = 0; i < n; i++) {

        // Bin point
        for (int k = 0; k < d; k++) {
            bins[i * d + k] = X_shifted[i * d + k] / bandwidth;
            current_bin[k] = bins[i * d + k];
        }

        if (means.find(current_bin) == means.end()) {
            means[current_bin] = make_pair(std::vector<float>(d, 0), 0);
        }
    }

    for (int i = 0; i < n; i++) {

        for (int j = 0; j < pow(base, d); j++) {

            // Get neighbor
            for (int k = 0; k < d; k++) {
                 current_bin[k] = bins[i * d + k] + offsets[j * d + k];
            }

            // If neighbor exists, add it to the mean
            if (means.find(current_bin) != means.end()) {

                for (int k = 0; k < d; k++) {
                    means[current_bin].first[k] += X_shifted[i * d + k];
                }

                means[current_bin].second++;
            }
        }
    }

    // Set every point to the mean of its neighbors
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