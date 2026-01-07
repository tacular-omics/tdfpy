use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Merge peaks using m/z and ion mobility tolerances.
///
/// Returns a tuple of three arrays: (mz_values, intensity_values, ion_mobility_values)
/// representing the merged peaks.
///
/// # Arguments
/// * `mz_array` - Array of m/z values
/// * `intensity_array` - Array of intensity values
/// * `ion_mobility_array` - Array of ion mobility values
/// * `mz_tolerance` - Tolerance for m/z matching
/// * `mz_tolerance_type` - Type of m/z tolerance: "ppm" or "da"
/// * `im_tolerance` - Tolerance for ion mobility matching
/// * `im_tolerance_type` - Type of ion mobility tolerance: "relative" or "absolute"
/// * `min_peaks` - Minimum number of nearby peaks required to form a centroid
/// * `max_peaks` - Optional maximum number of centroided peaks to return
#[pyfunction]
#[pyo3(signature = (mz_array, intensity_array, ion_mobility_array, mz_tolerance=8.0, mz_tolerance_type="ppm", im_tolerance=0.05, im_tolerance_type="relative", min_peaks=3, max_peaks=None))]
fn merge_peaks<'py>(
    py: Python<'py>,
    mz_array: PyReadonlyArray1<f64>,
    intensity_array: PyReadonlyArray1<f64>,
    ion_mobility_array: PyReadonlyArray1<f64>,
    mz_tolerance: f64,
    mz_tolerance_type: &str,
    im_tolerance: f64,
    im_tolerance_type: &str,
    min_peaks: usize,
    max_peaks: Option<usize>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let mz = mz_array.as_slice()?;
    let intensity = intensity_array.as_slice()?;
    let ion_mobility = ion_mobility_array.as_slice()?;

    let n = mz.len();
    if n == 0 {
        return Ok((
            PyArray1::zeros_bound(py, 0, false),
            PyArray1::zeros_bound(py, 0, false),
            PyArray1::zeros_bound(py, 0, false),
        ));
    }

    // Pre-compute tolerance factors
    let (mz_tol_factor, mz_tol_abs) = if mz_tolerance_type == "ppm" {
        (mz_tolerance / 1e6, 0.0)
    } else {
        (0.0, mz_tolerance)
    };

    let (mobility_tol_factor, mobility_tol_abs) = if im_tolerance_type == "relative" {
        (im_tolerance, 0.0)
    } else {
        (0.0, im_tolerance)
    };

    // Sort by m/z
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| mz[i].partial_cmp(&mz[j]).unwrap());

    let mut mz_sorted = vec![0.0; n];
    let mut intensity_sorted = vec![0.0; n];
    let mut ion_mobility_sorted = vec![0.0; n];

    for (new_idx, &old_idx) in indices.iter().enumerate() {
        mz_sorted[new_idx] = mz[old_idx];
        intensity_sorted[new_idx] = intensity[old_idx];
        ion_mobility_sorted[new_idx] = ion_mobility[old_idx];
    }

    // Sort by intensity for greedy clustering
    let mut intensity_order: Vec<usize> = (0..n).collect();
    intensity_order.sort_by(|&i, &j| {
        intensity_sorted[j]
            .partial_cmp(&intensity_sorted[i])
            .unwrap()
    });

    // Track used peaks
    let mut used = vec![false; n];
    let mut merged_mz = Vec::new();
    let mut merged_intensity = Vec::new();
    let mut merged_mobility = Vec::new();

    for &peak_idx in &intensity_order {
        if used[peak_idx] {
            continue;
        }

        let mz_peak = mz_sorted[peak_idx];
        let intensity_peak = intensity_sorted[peak_idx];
        let mobility_peak = ion_mobility_sorted[peak_idx];

        // Calculate tolerances
        let mz_tol = if mz_tolerance_type == "ppm" {
            mz_peak * mz_tol_factor
        } else {
            mz_tol_abs
        };

        let mobility_tol = if im_tolerance_type == "relative" {
            mobility_peak * mobility_tol_factor
        } else {
            mobility_tol_abs
        };

        // Binary search for m/z range
        let left_mz = mz_peak - mz_tol;
        let right_mz = mz_peak + mz_tol;

        let left_idx = mz_sorted.partition_point(|&x| x < left_mz);
        let right_idx = mz_sorted.partition_point(|&x| x <= right_mz);

        // Find nearby peaks in both m/z and mobility dimensions
        let mut nearby_indices = Vec::new();
        for i in left_idx..right_idx {
            if !used[i] && (ion_mobility_sorted[i] - mobility_peak).abs() <= mobility_tol {
                nearby_indices.push(i);
            }
        }

        let num_nearby = nearby_indices.len();

        // Check minimum peaks requirement
        if min_peaks > 0 && num_nearby < min_peaks {
            used[peak_idx] = true;
            continue;
        }

        if num_nearby == 0 {
            merged_mz.push(mz_peak);
            merged_intensity.push(intensity_peak);
            merged_mobility.push(mobility_peak);
            used[peak_idx] = true;
            continue;
        }

        // Centroid peaks using intensity-weighted average
        let mut total_intensity = 0.0;
        let mut weighted_mz = 0.0;
        let mut weighted_mobility = 0.0;

        for &idx in &nearby_indices {
            let int = intensity_sorted[idx];
            total_intensity += int;
            weighted_mz += mz_sorted[idx] * int;
            weighted_mobility += ion_mobility_sorted[idx] * int;
        }

        merged_mz.push(weighted_mz / total_intensity);
        merged_intensity.push(total_intensity);
        merged_mobility.push(weighted_mobility / total_intensity);

        // Mark as used
        for &idx in &nearby_indices {
            used[idx] = true;
        }

        if let Some(max) = max_peaks {
            if merged_mz.len() >= max {
                break;
            }
        }
    }

    // Convert to numpy arrays
    let result_mz = PyArray1::from_vec_bound(py, merged_mz);
    let result_intensity = PyArray1::from_vec_bound(py, merged_intensity);
    let result_mobility = PyArray1::from_vec_bound(py, merged_mobility);

    Ok((result_mz, result_intensity, result_mobility))
}

/// A Python module implemented in Rust.
#[pymodule]
fn _tdfpy_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(merge_peaks, m)?)?;
    Ok(())
}
