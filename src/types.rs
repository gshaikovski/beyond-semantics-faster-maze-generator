#[derive(Clone, Debug)]
pub struct Maze {
    pub grid: Vec<u8>,  // Bit-packed: each bit represents a cell (1=floor, 0=wall)
    pub start: (usize, usize),
    pub goal: (usize, usize),
    pub rows: usize,
    pub cols: usize,
    pub cols_bytes: usize,  // Number of bytes per row
}

impl Maze {
    /// Create a new maze with all walls
    pub fn new(rows: usize, cols: usize) -> Self {
        let cols_bytes = (cols + 7) / 8;
        Maze {
            grid: vec![0u8; rows * cols_bytes],
            start: (0, 0),
            goal: (0, 0),
            rows,
            cols,
            cols_bytes,
        }
    }
    
    /// Create a new maze with every cell a floor.
    ///
    /// The starting point for reading a maze back from a record: walls are punched into it as
    /// they are parsed, which is far cheaper than setting every cell.
    pub fn all_floor(rows: usize, cols: usize) -> Self {
        let mut maze = Maze::new(rows, cols);
        for y in 0..rows {
            for x in 0..cols {
                maze.set_cell(x, y, true);
            }
        }
        maze
    }

    /// Get cell value (true = floor, false = wall)
    #[inline(always)]
    pub fn get_cell(&self, x: usize, y: usize) -> bool {
        if x >= self.cols || y >= self.rows {
            return false;
        }
        let byte_idx = y * self.cols_bytes + (x / 8);
        let bit_idx = x % 8;
        (self.grid[byte_idx] & (1 << bit_idx)) != 0
    }
    
    /// Set cell value (true = floor, false = wall)
    #[inline(always)]
    pub fn set_cell(&mut self, x: usize, y: usize, value: bool) {
        if x >= self.cols || y >= self.rows {
            return;
        }
        let byte_idx = y * self.cols_bytes + (x / 8);
        let bit_idx = x % 8;
        if value {
            self.grid[byte_idx] |= 1 << bit_idx;
        } else {
            self.grid[byte_idx] &= !(1 << bit_idx);
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ReasoningEvent {
    Close { x: u16, y: u16, g: u16, h: u16 },
    Create { x: u16, y: u16, g: u16, h: u16 },
}

#[derive(Clone, Debug)]
pub struct Solution {
    pub path: Vec<(usize, usize)>,
    pub reasoning: Vec<ReasoningEvent>,
}

pub struct MazeResult {
    pub instance_id: u64,
    pub maze: Maze,
    pub solution: Solution,
    pub generator: crate::GeneratorType,
    pub solver: crate::SolverType,
    pub seed: u64,
}