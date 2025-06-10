#include <cmath>
#include <algorithm>

using namespace std;

void localshift_single_cy(float* point_pos,
                          int n_steps,
                          float fmaxd,
                          float fsiv,
                          float tol,
                          int ref_shape_0,
                          int ref_shape_1, 
                          int ref_shape_2,
                          float* reference) {
    /*
        Local shift algorithm for a single 3D point optimization.
        
        Parameters
        ----------
        point_pos: [3] array of point coordinates to be shifted (modified in-place)
        n_steps: Maximum number of iterations for this point
        fmaxd: Maximum distance for neighbor search
        fsiv: Kernel parameter for Gaussian weighting
        tol: Tolerance for convergence (squared distance)
        ref_shape_0, ref_shape_1, ref_shape_2: Dimensions of reference array
        reference: (ref_shape_0, ref_shape_1, ref_shape_2) reference array
    */
    
    float fsiv_neg = -1.5f * fsiv;
    
    for (int step = 0; step < n_steps; step++) {
        // Current position
        float pos[3] = {point_pos[0], point_pos[1], point_pos[2]};
        
        // Define search boundaries
        int stp[3] = {
            max(static_cast<int>(pos[0] - fmaxd), 0),
            max(static_cast<int>(pos[1] - fmaxd), 0),
            max(static_cast<int>(pos[2] - fmaxd), 0)
        };
        
        int endp[3] = {
            min(static_cast<int>(pos[0] + fmaxd + 1), ref_shape_0 - 1),
            min(static_cast<int>(pos[1] + fmaxd + 1), ref_shape_1 - 1),
            min(static_cast<int>(pos[2] + fmaxd + 1), ref_shape_2 - 1)
        };
        
        float pos2[3] = {0.0f, 0.0f, 0.0f};
        float dtotal = 0.0f;
        
        // Search in the 3D neighborhood
        for (int xp = stp[0]; xp < endp[0]; xp++) {
            for (int yp = stp[1]; yp < endp[1]; yp++) {
                for (int zp = stp[2]; zp < endp[2]; zp++) {
                    float offset[3] = {static_cast<float>(xp), static_cast<float>(yp), static_cast<float>(zp)};
                    
                    // Calculate squared distance
                    float d2 = (offset[0] - pos[0]) * (offset[0] - pos[0]) +
                              (offset[1] - pos[1]) * (offset[1] - pos[1]) +
                              (offset[2] - pos[2]) * (offset[2] - pos[2]);
                    
                    // Get reference value at this position
                    int ref_idx = xp * ref_shape_1 * ref_shape_2 + yp * ref_shape_2 + zp;
                    float kernel_weight = exp(fsiv_neg * d2) * reference[ref_idx];
                    
                    if (kernel_weight > 0) {
                        dtotal += kernel_weight;
                        pos2[0] += kernel_weight * offset[0];
                        pos2[1] += kernel_weight * offset[1];
                        pos2[2] += kernel_weight * offset[2];
                    }
                }
            }
        }
        
        if (dtotal > 0) {
            // Normalize to get new position
            pos2[0] /= dtotal;
            pos2[1] /= dtotal;
            pos2[2] /= dtotal;
            
            // Check convergence
            float shift_dis_square = (pos[0] - pos2[0]) * (pos[0] - pos2[0]) +
                                    (pos[1] - pos2[1]) * (pos[1] - pos2[1]) +
                                    (pos[2] - pos2[2]) * (pos[2] - pos2[2]);
            
            if (shift_dis_square < tol) {
                break;
            }
            
            // Update position
            point_pos[0] = pos2[0];
            point_pos[1] = pos2[1];
            point_pos[2] = pos2[2];
        }
    }
}
