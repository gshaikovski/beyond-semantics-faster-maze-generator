//! Layout-level de-duplication: what makes two maze instances "the same".
//!
//! Identity is the WALL LAYOUT — grid dimensions plus the packed floor/wall bitmap — and
//! ignores start and goal. Two instances sharing a layout share the same A* search graph and
//! differ only in which node the search enters and leaves from, so re-emitting one in another
//! split leaks a solved layout into eval. (This mirrors the countdown generator, which
//! excludes by number multiset whatever the target.)
//!
//! Under `--fixed-maze` every instance shares one layout by design, so identity is the QUERY
//! instead: the ordered (start, goal) pair.
//!
//! Stored keys are 128-bit FNV-1a digests, not the layouts themselves: 5M keys cost ~100MB
//! against ~400MB for verbatim 20x20 grids, and the birthday collision probability over 5M
//! keys is ~1e-25 — far below the chance of a disk error silently corrupting the dataset.

use crate::types::Maze;
use std::collections::HashSet;

/// A maze layout's 128-bit digest.
pub type LayoutKey = u128;

/// FNV-1a 128-bit offset basis and prime (Fowler/Noll/Vo, IETF draft-eastlake-fnv).
const FNV_OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
const FNV_PRIME: u128 = 0x0000000001000000000000000000013b;

/// FNV-1a over a byte slice.
fn fnv1a_128(seed: LayoutKey, bytes: &[u8]) -> LayoutKey {
    let mut hash = seed;
    for &byte in bytes {
        hash ^= byte as LayoutKey;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Layout key of a bit-packed grid.
///
/// Dimensions are folded in so grids of different shapes can never collide, and the caller's
/// packing must match `Maze` (row-major, `1 << (x % 8)` within `y * cols_bytes + x / 8`, unused
/// trailing bits zero).
pub fn key_from_parts(rows: usize, cols: usize, grid: &[u8]) -> LayoutKey {
    let mut hash = fnv1a_128(FNV_OFFSET, &(rows as u64).to_le_bytes());
    hash = fnv1a_128(hash, &(cols as u64).to_le_bytes());
    fnv1a_128(hash, grid)
}

/// Layout key of a generated maze (start and goal deliberately excluded).
pub fn layout_key(maze: &Maze) -> LayoutKey {
    key_from_parts(maze.rows, maze.cols, &maze.grid)
}

/// Query key of an instance on a fixed layout: its ordered (start, goal) pair.
/// - Packed losslessly (each coordinate < 2^32), so it cannot collide.
pub fn query_key(maze: &Maze) -> LayoutKey {
    let ((sx, sy), (gx, gy)) = (maze.start, maze.goal);
    [sx, sy, gx, gy]
        .iter()
        .fold(0, |key, &v| (key << 32) | v as LayoutKey)
}

/// The layouts already emitted (or excluded), shared across every partition of a run.
pub struct SeenSet {
    keys: HashSet<LayoutKey>,
}

impl SeenSet {
    /// Pre-size for `capacity` layouts; the set still grows on demand.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            keys: HashSet::with_capacity(capacity),
        }
    }

    /// Claim a layout. `true` when it is new (the caller may emit it), `false` when the layout
    /// was already emitted in this or an earlier partition, or pre-seeded by `--exclude`.
    pub fn claim(&mut self, key: LayoutKey) -> bool {
        self.keys.insert(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maze_with(rows: usize, cols: usize, walls: &[(usize, usize)]) -> Maze {
        let mut maze = Maze::new(rows, cols);
        for y in 0..rows {
            for x in 0..cols {
                maze.set_cell(x, y, true);
            }
        }
        for &(x, y) in walls {
            maze.set_cell(x, y, false);
        }
        maze
    }

    #[test]
    fn start_and_goal_do_not_change_the_key() {
        let mut a = maze_with(5, 5, &[(1, 1), (2, 3)]);
        let mut b = a.clone();
        a.start = (0, 0);
        a.goal = (4, 4);
        b.start = (3, 2);
        b.goal = (1, 0);
        assert_eq!(layout_key(&a), layout_key(&b));
    }

    #[test]
    fn a_single_flipped_cell_changes_the_key() {
        let a = maze_with(5, 5, &[(1, 1), (2, 3)]);
        let b = maze_with(5, 5, &[(1, 1), (2, 4)]);
        assert_ne!(layout_key(&a), layout_key(&b));
    }

    #[test]
    fn dimensions_are_part_of_the_key() {
        // Same packed bytes, different declared shape.
        let grid = vec![0u8; 4];
        assert_ne!(key_from_parts(2, 8, &grid), key_from_parts(4, 8, &grid));
        assert_ne!(key_from_parts(4, 8, &grid), key_from_parts(4, 5, &grid));
    }

    #[test]
    fn query_key_is_the_ordered_start_goal_pair() {
        let mut a = maze_with(5, 5, &[(1, 1)]);
        let mut b = maze_with(5, 5, &[(2, 2)]);
        (a.start, a.goal) = ((0, 1), (2, 3));
        (b.start, b.goal) = ((0, 1), (2, 3));
        assert_eq!(query_key(&a), query_key(&b));
        // Reversed query is a different instance (different trace and plan).
        b.start = (2, 3);
        b.goal = (0, 1);
        assert_ne!(query_key(&a), query_key(&b));
    }

    #[test]
    fn claim_is_idempotent_per_layout() {
        let mut seen = SeenSet::with_capacity(4);
        let key = layout_key(&maze_with(5, 5, &[(1, 1)]));
        assert!(seen.claim(key));
        assert!(!seen.claim(key));
        // A different layout is still claimable.
        assert!(seen.claim(layout_key(&maze_with(5, 5, &[(2, 2)]))));
    }
}
