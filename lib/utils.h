#include <map>
#include <queue>
#include <set>
#include <algorithm>

using namespace std;

void generate_offsets_cy(int d,
                         int base,
                         int * offsets) {
    /*
        Generate 3**d neighbors for any point.

        Parameters
        ----------
        d: Dimensions
        base: 3, corresponding to (-1, 0, 1)
        offsets: (3**d, d) array of offsets to be added to
                 a bin to get neighbors

    */

    int tmp_i;

    for (int i = 0; i < pow(base, d); i++) {
        tmp_i = i;
        for (int j = 0; j < d; j++) {
            if (tmp_i == 0) break;
            offsets[i * d + j] = tmp_i % base - 1;
            tmp_i /= base;
        }
    }
}
