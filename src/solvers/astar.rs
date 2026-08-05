use crate::types::{Maze, Solution, ReasoningEvent};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A* node for priority queue
#[derive(Copy, Clone, Eq, PartialEq)]
struct AStarNode {
    x: u16,
    y: u16,
    g_score: u16,
    f_score: u16,
}

impl Ord for AStarNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap based on f_score
        other.f_score.cmp(&self.f_score)
            .then_with(|| other.g_score.cmp(&self.g_score))
    }
}

impl PartialOrd for AStarNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Manhattan distance heuristic
#[inline(always)]
fn manhattan_distance(x1: u16, y1: u16, x2: u16, y2: u16) -> u16 {
    ((x1 as i32 - x2 as i32).abs() + (y1 as i32 - y2 as i32).abs()) as u16
}

/// A* pathfinding with reasoning trace
pub fn solve(maze: &Maze) -> Solution {
    let mut reasoning: Vec<ReasoningEvent> = Vec::with_capacity(1000);
    let mut open_set = BinaryHeap::with_capacity(256);
    
    // Use flat arrays for better cache locality
    let total_cells = maze.rows * maze.cols;
    let mut g_scores = vec![u16::MAX; total_cells];
    let mut came_from = vec![u32::MAX; total_cells];
    let mut closed_set = vec![false; total_cells];
    
    // Convert coordinates to u16 for efficiency
    let start_x = maze.start.0 as u16;
    let start_y = maze.start.1 as u16;
    let goal_x = maze.goal.0 as u16;
    let goal_y = maze.goal.1 as u16;
    let _cols = maze.cols as u16;
    
    // Initialize start node
    let start_idx = (start_y as usize) * maze.cols + (start_x as usize);
    let start_h = manhattan_distance(start_x, start_y, goal_x, goal_y);
    g_scores[start_idx] = 0;
    open_set.push(AStarNode {
        x: start_x,
        y: start_y,
        g_score: 0,
        f_score: start_h,
    });
    
    const DIRECTIONS: [(i16, i16); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];
    
    while let Some(current_node) = open_set.pop() {
        let x = current_node.x;
        let y = current_node.y;
        let current_idx = (y as usize) * maze.cols + (x as usize);
        
        // Skip if already processed
        if closed_set[current_idx] {
            continue;
        }
        
        let g_score = current_node.g_score;
        let h_score = manhattan_distance(x, y, goal_x, goal_y);
        
        // Record close event
        reasoning.push(ReasoningEvent::Close { 
            x: x, 
            y: y, 
            g: g_score, 
            h: h_score 
        });
        
        // Found goal
        if x == goal_x && y == goal_y {
            break;
        }
        
        closed_set[current_idx] = true;
        
        // Explore neighbors
        for &(dx, dy) in &DIRECTIONS {
            let nx = x as i16 + dx;
            let ny = y as i16 + dy;
            
            // Check bounds
            if nx < 0 || nx >= maze.cols as i16 || ny < 0 || ny >= maze.rows as i16 {
                continue;
            }
            
            let nx = nx as u16;
            let ny = ny as u16;
            let neighbor_idx = (ny as usize) * maze.cols + (nx as usize);
            
            // Skip walls and closed nodes
            if !maze.get_cell(nx as usize, ny as usize) || closed_set[neighbor_idx] {
                continue;
            }
            
            let tentative_g = g_score + 1;
            
            // Update if this is a better path
            if tentative_g < g_scores[neighbor_idx] {
                came_from[neighbor_idx] = current_idx as u32;
                g_scores[neighbor_idx] = tentative_g;
                
                let h = manhattan_distance(nx, ny, goal_x, goal_y);
                let f = tentative_g + h;
                
                // Record create event
                reasoning.push(ReasoningEvent::Create { 
                    x: nx, 
                    y: ny, 
                    g: tentative_g, 
                    h 
                });
                
                open_set.push(AStarNode {
                    x: nx,
                    y: ny,
                    g_score: tentative_g,
                    f_score: f,
                });
            }
        }
    }
    
    // Reconstruct path
    let mut path = Vec::with_capacity(100);
    let goal_idx = (goal_y as usize) * maze.cols + (goal_x as usize);
    let mut current_idx = goal_idx;
    
    // Check if goal was reached
    if came_from[goal_idx] != u32::MAX || current_idx == start_idx {
        while current_idx != start_idx {
            let x = (current_idx % maze.cols) as usize;
            let y = (current_idx / maze.cols) as usize;
            path.push((x, y));
            
            let prev_idx = came_from[current_idx];
            if prev_idx == u32::MAX {
                // No path found
                path.clear();
                break;
            }
            current_idx = prev_idx as usize;
        }
        
        if !path.is_empty() {
            path.push(maze.start);
            path.reverse();
        }
    }

    Solution { path, reasoning }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::collections::{HashMap, HashSet, VecDeque};

    /// Build a maze from an ASCII map: `#` wall, `S` start, `G` goal, anything else floor.
    fn maze_from(map: &[&str]) -> Maze {
        let rows = map.len();
        let cols = map[0].len();
        let mut maze = Maze::new(rows, cols);
        for (y, line) in map.iter().enumerate() {
            assert_eq!(line.len(), cols, "ragged map");
            for (x, ch) in line.chars().enumerate() {
                maze.set_cell(x, y, ch != '#');
                match ch {
                    'S' => maze.start = (x, y),
                    'G' => maze.goal = (x, y),
                    _ => {}
                }
            }
        }
        maze
    }

    /// True shortest-path distances from `from`, by BFS: ground truth independent of A*.
    fn bfs(maze: &Maze, from: (usize, usize)) -> HashMap<(usize, usize), usize> {
        let mut dist = HashMap::from([(from, 0usize)]);
        let mut queue = VecDeque::from([from]);
        while let Some((x, y)) = queue.pop_front() {
            let d = dist[&(x, y)];
            for (dx, dy) in [(0i32, -1i32), (1, 0), (0, 1), (-1, 0)] {
                let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                if nx < 0 || ny < 0 || nx >= maze.cols as i32 || ny >= maze.rows as i32 {
                    continue;
                }
                let next = (nx as usize, ny as usize);
                if !maze.get_cell(next.0, next.1) || dist.contains_key(&next) {
                    continue;
                }
                dist.insert(next, d + 1);
                queue.push_back(next);
            }
        }
        dist
    }

    /// Check a solution against BFS ground truth.
    ///
    /// The properties that make the search sound on a graph with cycles, where a cell can be
    /// reached by several paths:
    /// - no cell is expanded (`close`d) twice — the closed set actually holds;
    /// - every `close`/`create` carries the cell's true distance from the start, so no cell is
    ///   ever settled on a worse path than one already found;
    /// - the emitted plan is a shortest path.
    fn assert_sound(maze: &Maze) {
        let solution = solve(maze);
        let dist = bfs(maze, maze.start);

        let mut closed: HashSet<(u16, u16)> = HashSet::new();
        let mut created: HashSet<(u16, u16)> = HashSet::new();
        for event in &solution.reasoning {
            let (kind, x, y, g, h) = match *event {
                ReasoningEvent::Close { x, y, g, h } => ("close", x, y, g, h),
                ReasoningEvent::Create { x, y, g, h } => ("create", x, y, g, h),
            };
            let cell = (x as usize, y as usize);
            assert!(maze.get_cell(cell.0, cell.1), "{kind} on a wall at {cell:?}");
            assert_eq!(
                Some(g as usize),
                dist.get(&cell).copied(),
                "{kind} {cell:?} carries g={g}, true distance {:?}",
                dist.get(&cell)
            );
            assert_eq!(
                h as usize,
                cell.0.abs_diff(maze.goal.0) + cell.1.abs_diff(maze.goal.1),
                "{kind} {cell:?} carries a wrong h"
            );
            let first = match kind {
                "close" => closed.insert((x, y)),
                _ => created.insert((x, y)),
            };
            assert!(first, "{cell:?} {kind}d twice");
        }

        match dist.get(&maze.goal) {
            None => assert!(solution.path.is_empty(), "path to an unreachable goal"),
            Some(&d) => {
                assert_eq!(solution.path.first(), Some(&maze.start));
                assert_eq!(solution.path.last(), Some(&maze.goal));
                assert_eq!(solution.path.len(), d + 1, "path is not a shortest path");
                for pair in solution.path.windows(2) {
                    let (a, b) = (pair[0], pair[1]);
                    assert_eq!(a.0.abs_diff(b.0) + a.1.abs_diff(b.1), 1, "path jumps");
                    assert!(maze.get_cell(b.0, b.1), "path crosses a wall");
                }
            }
        }
    }

    #[test]
    fn wall_free_grid_is_solved_optimally() {
        // Maximally looped: every interior cell has four ways in.
        assert_sound(&maze_from(&[
            "S.......", "........", "........", "........", "........", "........", "........",
            ".......G",
        ]));
    }

    #[test]
    fn two_paths_of_equal_length_round_a_loop() {
        // A single ring: the goal is reachable clockwise and anticlockwise at the same cost.
        assert_sound(&maze_from(&["S....", ".###.", ".###.", ".###.", "....G"]));
    }

    #[test]
    fn the_short_way_round_leads_away_from_the_goal() {
        // The Manhattan-favoured corridor to the right is a long detour; the shortest route
        // starts by moving away from the goal, so cells are reached from both sides.
        assert_sound(&maze_from(&[
            "..........",
            ".########.",
            ".#S#####G.",
            ".#.#####.#",
            ".#.......#",
            ".#########",
        ]));
    }

    #[test]
    fn goal_behind_a_sealed_wall_has_no_path() {
        let maze = maze_from(&["S..#..G", "...#...", "...#..."]);
        assert_sound(&maze);
        assert!(solve(&maze).path.is_empty());
    }

    #[test]
    fn random_looped_mazes_agree_with_bfs() {
        // Random wall fields are riddled with cycles (unlike the spanning-tree generators), so
        // this is the case the closed set has to get right.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(20260805);
        let (rows, cols) = (12, 12);
        let mut solvable = 0;
        let mut looped = 0;
        for _ in 0..500 {
            let mut maze = Maze::new(rows, cols);
            for y in 0..rows {
                for x in 0..cols {
                    maze.set_cell(x, y, rng.gen_bool(0.75));
                }
            }
            maze.start = (0, 0);
            maze.goal = (cols - 1, rows - 1);
            maze.set_cell(maze.start.0, maze.start.1, true);
            maze.set_cell(maze.goal.0, maze.goal.1, true);
            assert_sound(&maze);
            let dist = bfs(&maze, maze.start);
            solvable += dist.contains_key(&maze.goal) as usize;
            // Cycles exist iff the reachable component has more edges than a spanning tree.
            let edges: usize = dist
                .keys()
                .map(|&(x, y)| {
                    [(1i32, 0i32), (0, 1)]
                        .iter()
                        .filter(|(dx, dy)| {
                            dist.contains_key(&((x as i32 + dx) as usize, (y as i32 + dy) as usize))
                        })
                        .count()
                })
                .sum();
            looped += (edges + 1 > dist.len()) as usize;
        }
        // Guard the fixture itself: the sample has to actually contain solvable, looped mazes.
        assert!(solvable > 100, "only {solvable}/500 solvable");
        assert!(looped > 400, "only {looped}/500 had cycles");
    }
}