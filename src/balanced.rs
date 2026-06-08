//! Helpers for balanced (uniform plan-length) sampling.
//!
//! Plan length = number of cells in the A* solution path (== downstream `plan_len`).
//! These are pure functions so the binning / cap logic is unit-testable without threads.

/// Map a plan length to an equal-width bin index.
/// - Returns `None` if `len` is outside `[min_len, max_len]` or inputs are degenerate.
/// - Bins partition `[min_len, max_len]` into `num_bins` equal-width buckets.
pub fn bin_index(len: usize, min_len: usize, max_len: usize, num_bins: usize) -> Option<usize> {
    if num_bins == 0 || max_len < min_len || len < min_len || len > max_len {
        return None;
    }
    let width = max_len - min_len + 1;
    let bin = (len - min_len) * num_bins / width;
    // Belt-and-suspenders: never exceeds num_bins-1 when width >= num_bins.
    Some(bin.min(num_bins - 1))
}

/// Per-bin acceptance caps summing exactly to `count`.
/// - Distributes the remainder across the first `count % num_bins` bins.
pub fn compute_caps(count: u64, num_bins: usize) -> Vec<u64> {
    if num_bins == 0 {
        return Vec::new();
    }
    let n = num_bins as u64;
    let base = count / n;
    let rem = count % n;
    (0..num_bins)
        .map(|b| base + if (b as u64) < rem { 1 } else { 0 })
        .collect()
}

/// Clamp `num_bins` to the number of integer lengths in `[min_len, max_len]`.
/// - A bin with no integer length can never fill, which would deadlock the loop.
pub fn clamp_num_bins(num_bins: usize, min_len: usize, max_len: usize) -> usize {
    let width = max_len.saturating_sub(min_len) + 1;
    num_bins.min(width).max(1)
}

/// Nearest-rank percentile of a set of lengths, used to trim the calibrated max.
/// - `q` is clamped to `[0, 1]`; returns `None` for an empty slice.
pub fn percentile(values: &[usize], q: f64) -> Option<usize> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let q = q.clamp(0.0, 1.0);
    let n = sorted.len();
    // Nearest-rank: rank in [1, n].
    let rank = (q * n as f64).ceil() as usize;
    let idx = rank.clamp(1, n) - 1;
    Some(sorted[idx])
}

/// Inclusive `[lo, hi]` plan-length range covered by each bin.
/// - Used only for the under-fill report; empty bins map to `None`.
pub fn bin_ranges(min_len: usize, max_len: usize, num_bins: usize) -> Vec<Option<(usize, usize)>> {
    let mut ranges: Vec<Option<(usize, usize)>> = vec![None; num_bins];
    if max_len < min_len {
        return ranges;
    }
    for len in min_len..=max_len {
        if let Some(b) = bin_index(len, min_len, max_len, num_bins) {
            ranges[b] = match ranges[b] {
                None => Some((len, len)),
                Some((lo, hi)) => Some((lo.min(len), hi.max(len))),
            };
        }
    }
    ranges
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_index_endpoints() {
        // First length -> bin 0, last length -> last bin.
        assert_eq!(bin_index(5, 5, 14, 5), Some(0));
        assert_eq!(bin_index(14, 5, 14, 5), Some(4));
    }

    #[test]
    fn bin_index_out_of_range() {
        assert_eq!(bin_index(4, 5, 14, 5), None);
        assert_eq!(bin_index(15, 5, 14, 5), None);
        // Empty / unsolvable path.
        assert_eq!(bin_index(0, 5, 14, 5), None);
    }

    #[test]
    fn bin_index_monotone_and_in_range() {
        // Sweeping lengths yields non-decreasing bins, all within [0, num_bins).
        let (min_len, max_len, num_bins) = (3usize, 30usize, 7usize);
        let mut prev = 0usize;
        for len in min_len..=max_len {
            let b = bin_index(len, min_len, max_len, num_bins).unwrap();
            assert!(b < num_bins);
            assert!(b >= prev);
            prev = b;
        }
    }

    #[test]
    fn bin_index_all_bins_reachable() {
        // When width >= num_bins, every bin is hit by some length.
        let (min_len, max_len, num_bins) = (1usize, 100usize, 10usize);
        let mut hit = vec![false; num_bins];
        for len in min_len..=max_len {
            hit[bin_index(len, min_len, max_len, num_bins).unwrap()] = true;
        }
        assert!(hit.iter().all(|&h| h));
    }

    #[test]
    fn caps_sum_to_count() {
        let caps = compute_caps(1000, 7);
        assert_eq!(caps.iter().sum::<u64>(), 1000);
        // Caps differ by at most 1, first `rem` bins get the +1.
        let max = *caps.iter().max().unwrap();
        let min = *caps.iter().min().unwrap();
        assert!(max - min <= 1);
        assert_eq!(caps[0], max);
    }

    #[test]
    fn caps_count_smaller_than_bins() {
        // count < num_bins: first `count` bins get 1, rest get 0 (a 0-cap bin is trivially full).
        let caps = compute_caps(3, 5);
        assert_eq!(caps, vec![1, 1, 1, 0, 0]);
        assert_eq!(caps.iter().sum::<u64>(), 3);
    }

    #[test]
    fn clamp_num_bins_narrow_range() {
        // Range [5, 7] has 3 integer lengths -> at most 3 bins.
        assert_eq!(clamp_num_bins(10, 5, 7), 3);
        // Wide range is untouched.
        assert_eq!(clamp_num_bins(10, 1, 100), 10);
        // Never returns 0.
        assert_eq!(clamp_num_bins(0, 5, 5), 1);
    }

    #[test]
    fn percentile_basic() {
        let vals: Vec<usize> = (1..=100).collect();
        assert_eq!(percentile(&vals, 1.0), Some(100));
        assert_eq!(percentile(&vals, 0.99), Some(99));
        assert_eq!(percentile(&vals, 0.0), Some(1));
        assert_eq!(percentile(&[], 0.99), None);
    }

    #[test]
    fn bin_ranges_partition() {
        // Ranges are contiguous, non-overlapping, and cover [min_len, max_len].
        let (min_len, max_len, num_bins) = (1usize, 100usize, 10usize);
        let ranges = bin_ranges(min_len, max_len, num_bins);
        assert_eq!(ranges.len(), num_bins);
        let first = ranges[0].unwrap();
        let last = ranges[num_bins - 1].unwrap();
        assert_eq!(first.0, min_len);
        assert_eq!(last.1, max_len);
        for w in ranges.windows(2) {
            let (_, hi) = w[0].unwrap();
            let (lo, _) = w[1].unwrap();
            assert_eq!(lo, hi + 1);
        }
    }
}
