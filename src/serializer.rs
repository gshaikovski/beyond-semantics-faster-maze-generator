use crate::types::{MazeResult, ReasoningEvent};
use crate::{GeneratorType, SolverType};
use rayon::prelude::*;
use std::fmt::Write as FmtWrite;
use std::io::Write;

/// Write a maze result directly to a writer as JSON
pub fn write_maze_json<W: Write>(
    writer: &mut W, 
    result: &MazeResult,
    buffer: &mut String,  // Reusable buffer for number formatting
) -> std::io::Result<()> {
    // Write opening brace
    writer.write_all(b"{")?;
    
    // Write idx field
    writer.write_all(b"\"idx\":")?;
    buffer.clear();
    write!(buffer, "{}", result.instance_id).unwrap();
    writer.write_all(buffer.as_bytes())?;
    
    // Write text field with maze data
    writer.write_all(b",\"text\":\"query start ")?;
    
    // Write coordinates using the buffer
    buffer.clear();
    write!(buffer, "{} {} goal {} {}", 
        result.maze.start.0, result.maze.start.1,
        result.maze.goal.0, result.maze.goal.1).unwrap();
    writer.write_all(buffer.as_bytes())?;
    
    // Write walls directly
    for y in 0..result.maze.rows {
        for x in 0..result.maze.cols {
            if !result.maze.get_cell(x, y) {
                writer.write_all(b" wall ")?;
                buffer.clear();
                write!(buffer, "{} {}", x, y).unwrap();
                writer.write_all(buffer.as_bytes())?;
            }
        }
    }
    
    // Write reasoning trace
    writer.write_all(b" reasoning")?;
    for event in &result.solution.reasoning {
        match event {
            ReasoningEvent::Close { x, y, g, h } => {
                writer.write_all(b" close ")?;
                buffer.clear();
                write!(buffer, "{} {} c{} c{}", x, y, g, h).unwrap();
                writer.write_all(buffer.as_bytes())?;
            }
            ReasoningEvent::Create { x, y, g, h } => {
                writer.write_all(b" create ")?;
                buffer.clear();
                write!(buffer, "{} {} c{} c{}", x, y, g, h).unwrap();
                writer.write_all(buffer.as_bytes())?;
            }
        }
    }
    
    // Write solution path
    writer.write_all(b" solution")?;
    for &(x, y) in &result.solution.path {
        writer.write_all(b" plan ")?;
        buffer.clear();
        write!(buffer, "{} {}", x, y).unwrap();
        writer.write_all(buffer.as_bytes())?;
    }
    
    writer.write_all(b" end\"")?;
    
    // Write remaining fields
    writer.write_all(b",\"generator\":\"")?;
    match result.generator {
        GeneratorType::Dfs => writer.write_all(b"dfs")?,
        GeneratorType::Kruskal => writer.write_all(b"kruskal")?,
        GeneratorType::Wilson => writer.write_all(b"wilson")?,
        GeneratorType::Searchformer => writer.write_all(b"searchformer")?,
        GeneratorType::DrunkardsWalk => writer.write_all(b"drunkardswalk")?,
    }
    
    writer.write_all(b"\",\"solver\":\"")?;
    match result.solver {
        SolverType::AStar => writer.write_all(b"astar")?,
    }
    
    writer.write_all(b"\",\"seed\":")?;
    buffer.clear();
    write!(buffer, "{}", result.seed).unwrap();
    writer.write_all(buffer.as_bytes())?;
    
    writer.write_all(b",\"rows\":")?;
    buffer.clear();
    write!(buffer, "{}", result.maze.rows).unwrap();
    writer.write_all(buffer.as_bytes())?;
    
    writer.write_all(b",\"cols\":")?;
    buffer.clear();
    write!(buffer, "{}", result.maze.cols).unwrap();
    writer.write_all(buffer.as_bytes())?;
    
    writer.write_all(b"}\n")?;
    
    Ok(())
}

// Thread-local string buffer pool
thread_local! {
    static FORMAT_BUFFER: std::cell::RefCell<String> = std::cell::RefCell::new(String::with_capacity(256));
}


/// Serialize a contiguous run of maze results into one byte buffer.
fn serialize_run(results: &[MazeResult]) -> Vec<u8> {
    FORMAT_BUFFER.with(|buf_cell| {
        let mut buffer = buf_cell.borrow_mut();
        // Use 8KB per maze
        let mut output = Vec::with_capacity(results.len() * 8192);

        for result in results {
            write_maze_json(&mut output, result, &mut buffer).unwrap();
        }

        output
    })
}

/// Sub-chunk size for parallel serialization; small enough to spread a batch over the pool.
const SERIALIZE_CHUNK: usize = 32;

/// Process a batch of maze results and write them to a byte vector.
///
/// Records are accepted serially (that is what makes the output deterministic), so formatting
/// them there would idle every other core on a large run. Sub-chunks are serialized on the
/// rayon pool and concatenated in order, which keeps the bytes identical either way.
pub fn process_batch(results: &[MazeResult]) -> Vec<u8> {
    if results.len() <= SERIALIZE_CHUNK {
        return serialize_run(results);
    }
    let chunks: Vec<Vec<u8>> = results
        .par_chunks(SERIALIZE_CHUNK)
        .map(serialize_run)
        .collect();
    chunks.concat()
}