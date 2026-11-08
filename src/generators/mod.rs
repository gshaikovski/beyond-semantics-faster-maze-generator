pub mod wilson;
pub mod dfs;
pub mod kruskal;
pub mod drunkards_walk;
pub mod searchformer;

use crate::types::Maze;
use crate::GeneratorType;
use crate::parameters::GeneratorParams;
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub fn generate_maze(
    generator: GeneratorType,
    rng: &mut Xoshiro256PlusPlus,
    rows: usize,
    cols: usize,
    params: &GeneratorParams,
) -> Maze {
    match generator {
        GeneratorType::Wilson => wilson::generate(rng, rows, cols),
        GeneratorType::Dfs => dfs::generate(rng, rows, cols),
        GeneratorType::Kruskal => kruskal::generate(rng, rows, cols),
        GeneratorType::DrunkardsWalk => drunkards_walk::generate(rng, rows, cols, params),
        GeneratorType::Searchformer => searchformer::generate(rng, rows, cols),
    }
}
/// Floor cells of a maze, row-major.
pub fn floors(maze: &Maze) -> Vec<(usize, usize)> {
    (0..maze.rows)
        .flat_map(|y| (0..maze.cols).map(move |x| (x, y)))
        .filter(|&(x, y)| maze.get_cell(x, y))
        .collect()
}

/// A copy of `layout` with a fresh query: start/goal uniform over ordered pairs of distinct floor
/// cells (`--fixed-maze`).
pub fn with_random_query(layout: &Maze, rng: &mut Xoshiro256PlusPlus) -> Maze {
    let floors = floors(layout);
    let start = rng.gen_range(0..floors.len());
    // Uniform over the other n-1 cells: draw an index in 0..n-1 and step over `start`.
    let mut goal = rng.gen_range(0..floors.len() - 1);
    if goal >= start {
        goal += 1;
    }
    let mut maze = layout.clone();
    maze.start = floors[start];
    maze.goal = floors[goal];
    maze
}
