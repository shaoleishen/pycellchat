/// Spatial distance computation using grid-based spatial hashing.
///
/// For 2D coordinates, this provides O(N) average-case neighbor search
/// instead of O(N²) brute-force pairwise distance.

use sprs::{CsMat, TriMat};
use std::collections::HashMap;

/// Compute sparse cell-cell distance matrix using grid-based spatial hashing.
///
/// Only computes distances within `interaction_range`. Returns sparse matrices.
///
/// # Arguments
/// * `coordinates` - N x 2 array of (x, y) coordinates
/// * `interaction_range` - maximum distance to consider (in coordinate units)
/// * `contact_range` - distance threshold for contact-dependent signaling
/// * `ratio` - pixel-to-unit conversion factor (multiplied with distances)
/// * `tol` - distance tolerance added to thresholds
///
/// # Returns
/// (d_spatial, adj_contact) as sparse CSR matrices
pub fn compute_sparse_cell_distance(
    coordinates: &[Vec<f64>],
    interaction_range: f64,
    contact_range: f64,
    ratio: f64,
    tol: f64,
) -> (CsMat<f64>, CsMat<f64>) {
    let n_cells = coordinates.len();
    if n_cells == 0 {
        let empty = TriMat::new((0, 0));
        return (empty.to_csr(), TriMat::new((0, 0)).to_csr());
    }

    let threshold = interaction_range + tol;
    let contact_threshold = contact_range + tol;

    // Grid cell size = threshold (so we only need to check 3x3 neighborhood)
    let cell_size = threshold;
    if cell_size <= 0.0 {
        let empty = TriMat::new((n_cells, n_cells));
        return (empty.to_csr(), TriMat::new((n_cells, n_cells)).to_csr());
    }

    // Build spatial grid: map (gx, gy) -> list of cell indices
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (idx, coord) in coordinates.iter().enumerate() {
        let gx = (coord[0] / cell_size).floor() as i64;
        let gy = (coord[1] / cell_size).floor() as i64;
        grid.entry((gx, gy)).or_default().push(idx);
    }

    // For each cell, check 3x3 neighborhood grid cells
    let mut dist_triplet = TriMat::new((n_cells, n_cells));
    let mut contact_triplet = TriMat::new((n_cells, n_cells));

    for (idx, coord) in coordinates.iter().enumerate() {
        let gx = (coord[0] / cell_size).floor() as i64;
        let gy = (coord[1] / cell_size).floor() as i64;

        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(neighbors) = grid.get(&(gx + dx, gy + dy)) {
                    for &j in neighbors {
                        if j <= idx {
                            continue; // skip self and already-processed pairs
                        }
                        let dx_val = coord[0] - coordinates[j][0];
                        let dy_val = coord[1] - coordinates[j][1];
                        let dist = (dx_val * dx_val + dy_val * dy_val).sqrt() * ratio;

                        if dist > 0.0 && dist <= threshold {
                            dist_triplet.add_triplet(idx, j, dist);
                            dist_triplet.add_triplet(j, idx, dist);

                            if dist <= contact_threshold {
                                contact_triplet.add_triplet(idx, j, 1.0);
                                contact_triplet.add_triplet(j, idx, 1.0);
                            }
                        }
                    }
                }
            }
        }
    }

    (dist_triplet.to_csr(), contact_triplet.to_csr())
}

/// Convert sparse distance matrix to sparse P_spatial.
/// P_spatial(i,j) = 1 / (d(i,j) * scale_distance)
/// Diagonal set to max(P_spatial) for autocrine signaling.
pub fn create_p_spatial_from_distance(
    d_spatial: &CsMat<f64>,
    scale_distance: f64,
) -> CsMat<f64> {
    let n = d_spatial.rows();
    let mut p_triplet = TriMat::new((n, n));

    let mut max_val = 0.0_f64;

    // Process non-zero entries
    for (row_idx, row) in d_spatial.outer_iterator().enumerate() {
        for (col_idx, &dist) in row.iter() {
            if dist > 0.0 && row_idx != col_idx {
                let val = 1.0 / (dist * scale_distance);
                p_triplet.add_triplet(row_idx, col_idx, val);
                if val > max_val {
                    max_val = val;
                }
            }
        }
    }

    // Set diagonal to max value (autocrine signaling)
    for i in 0..n {
        p_triplet.add_triplet(i, i, max_val);
    }

    p_triplet.to_csr()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_distance_basic() {
        // 4 points in a line
        let coords = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![3.0, 0.0],
            vec![10.0, 0.0],
        ];
        let (dist, contact) = compute_sparse_cell_distance(&coords, 5.0, 1.5, 1.0, 0.0);

        // Points 0-1: distance 1.0 (within range)
        // Points 0-2: distance 3.0 (within range)
        // Points 0-3: distance 10.0 (out of range)
        // Points 1-2: distance 2.0 (within range)
        // Points 1-3: distance 9.0 (out of range)
        // Points 2-3: distance 7.0 (out of range)

        assert_eq!(dist.rows(), 4);
        assert_eq!(dist.cols(), 4);

        // Check specific entries
        let d01 = dist.get(0, 1).copied().unwrap_or(0.0);
        assert!((d01 - 1.0).abs() < 1e-10);

        let d03 = dist.get(0, 3).copied().unwrap_or(0.0);
        assert_eq!(d03, 0.0); // out of range

        // Contact adjacency: only pairs within 1.5
        let c01 = contact.get(0, 1).copied().unwrap_or(0.0);
        assert_eq!(c01, 1.0); // distance 1.0 <= 1.5

        let c02 = contact.get(0, 2).copied().unwrap_or(0.0);
        assert_eq!(c02, 0.0); // distance 3.0 > 1.5
    }

    #[test]
    fn test_sparse_distance_symmetric() {
        let coords = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let (dist, _) = compute_sparse_cell_distance(&coords, 5.0, 1.5, 1.0, 0.0);

        let d01 = dist.get(0, 1).copied().unwrap_or(0.0);
        let d10 = dist.get(1, 0).copied().unwrap_or(0.0);
        assert!((d01 - d10).abs() < 1e-10);
    }

    #[test]
    fn test_p_spatial() {
        let coords = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
        ];
        let (dist, _) = compute_sparse_cell_distance(&coords, 5.0, 1.5, 1.0, 0.0);
        let p = create_p_spatial_from_distance(&dist, 0.1);

        // Diagonal should be max(P_spatial)
        let diag = p.get(0, 0).copied().unwrap_or(0.0);
        assert!(diag > 0.0);

        // P(0,1) = 1/(1.0 * 0.1) = 10.0
        let p01 = p.get(0, 1).copied().unwrap_or(0.0);
        assert!((p01 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_input() {
        let coords: Vec<Vec<f64>> = vec![];
        let (dist, contact) = compute_sparse_cell_distance(&coords, 5.0, 1.5, 1.0, 0.0);
        assert_eq!(dist.rows(), 0);
        assert_eq!(contact.rows(), 0);
    }
}
