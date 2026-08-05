use crate::types::{MazeResult, ReasoningEvent, Solution};
use crate::{GeneratorType, SolverType};
use rayon::prelude::*;
use std::fmt::Write as FmtWrite;
use std::io::Write;

/// Where the `query start ... goal ...` block sits inside `text`.
///
/// The token multiset is identical either way; only the order changes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QueryPlacement {
    /// Default: `wall ... query start sx sy goal gx gy reasoning ...`.
    /// The query is adjacent to the trace it conditions, so the maze body cannot push it out
    /// of a model's effective context.
    #[default]
    AfterWalls,
    /// `--legacy-query-first`: the original layout,
    /// `query start sx sy goal gx gy wall ... reasoning ...`.
    First,
}

/// Write the `query start sx sy goal gx gy` block.
fn write_query_block<W: Write>(
    writer: &mut W,
    result: &MazeResult,
    buffer: &mut String,
) -> std::io::Result<()> {
    writer.write_all(b"query start ")?;
    buffer.clear();
    write!(
        buffer,
        "{} {} goal {} {}",
        result.maze.start.0, result.maze.start.1, result.maze.goal.0, result.maze.goal.1
    )
    .unwrap();
    writer.write_all(buffer.as_bytes())
}

/// Write the `wall x y` segments in row-major order.
///
/// - `trailing`: put the separating space *after* each segment rather than before it. Walls
///   open `text` in the default placement and follow the query block in the legacy one, so the
///   space has to sit on the side that faces the query block either way.
fn write_walls<W: Write>(
    writer: &mut W,
    result: &MazeResult,
    buffer: &mut String,
    trailing: bool,
) -> std::io::Result<()> {
    for y in 0..result.maze.rows {
        for x in 0..result.maze.cols {
            if result.maze.get_cell(x, y) {
                continue;
            }
            if !trailing {
                writer.write_all(b" ")?;
            }
            writer.write_all(b"wall ")?;
            buffer.clear();
            write!(buffer, "{} {}", x, y).unwrap();
            writer.write_all(buffer.as_bytes())?;
            if trailing {
                writer.write_all(b" ")?;
            }
        }
    }
    Ok(())
}

/// Write the trace and plan half of a `text` value: ` reasoning … solution … end`.
///
/// Stops short of the closing quote, so `--puzzle-file` can splice this onto a record's
/// original prompt bytes and follow it with the rest of that record verbatim.
pub fn write_trace_and_plan<W: Write>(
    writer: &mut W,
    solution: &Solution,
    buffer: &mut String,
) -> std::io::Result<()> {
    writer.write_all(b" reasoning")?;
    for event in &solution.reasoning {
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

    writer.write_all(b" solution")?;
    for &(x, y) in &solution.path {
        writer.write_all(b" plan ")?;
        buffer.clear();
        write!(buffer, "{} {}", x, y).unwrap();
        writer.write_all(buffer.as_bytes())?;
    }

    writer.write_all(b" end")
}

/// Write a maze result directly to a writer as JSON
pub fn write_maze_json<W: Write>(
    writer: &mut W,
    result: &MazeResult,
    // Reusable buffer for number formatting
    buffer: &mut String,
    placement: QueryPlacement,
) -> std::io::Result<()> {
    // Write opening brace
    writer.write_all(b"{")?;

    // Write idx field
    writer.write_all(b"\"idx\":")?;
    buffer.clear();
    write!(buffer, "{}", result.instance_id).unwrap();
    writer.write_all(buffer.as_bytes())?;

    // Write text field with maze data
    writer.write_all(b",\"text\":\"")?;

    // Prompt: the query block and the wall list, in the requested order.
    match placement {
        QueryPlacement::AfterWalls => {
            write_walls(writer, result, buffer, true)?;
            write_query_block(writer, result, buffer)?;
        }
        QueryPlacement::First => {
            write_query_block(writer, result, buffer)?;
            write_walls(writer, result, buffer, false)?;
        }
    }

    // Write the reasoning trace and the solution path
    write_trace_and_plan(writer, &result.solution, buffer)?;

    // Close the text value
    writer.write_all(b"\"")?;

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
fn serialize_run(results: &[MazeResult], placement: QueryPlacement) -> Vec<u8> {
    FORMAT_BUFFER.with(|buf_cell| {
        let mut buffer = buf_cell.borrow_mut();
        // Use 8KB per maze
        let mut output = Vec::with_capacity(results.len() * 8192);

        for result in results {
            write_maze_json(&mut output, result, &mut buffer, placement).unwrap();
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
pub fn process_batch(results: &[MazeResult], placement: QueryPlacement) -> Vec<u8> {
    if results.len() <= SERIALIZE_CHUNK {
        return serialize_run(results, placement);
    }
    let chunks: Vec<Vec<u8>> = results
        .par_chunks(SERIALIZE_CHUNK)
        .map(|chunk| serialize_run(chunk, placement))
        .collect();
    chunks.concat()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Maze;

    /// A 3x3 maze with walls at (1,0) and (0,2), solved by a two-step trace.
    fn result_3x3(walls: &[(usize, usize)]) -> MazeResult {
        let mut maze = Maze::new(3, 3);
        for y in 0..3 {
            for x in 0..3 {
                maze.set_cell(x, y, true);
            }
        }
        for &(x, y) in walls {
            maze.set_cell(x, y, false);
        }
        maze.start = (0, 0);
        maze.goal = (2, 2);
        MazeResult {
            instance_id: 7,
            maze,
            solution: Solution {
                path: vec![(0, 0), (2, 2)],
                reasoning: vec![
                    ReasoningEvent::Create {
                        x: 0,
                        y: 0,
                        g: 0,
                        h: 4,
                    },
                    ReasoningEvent::Close {
                        x: 0,
                        y: 0,
                        g: 0,
                        h: 4,
                    },
                ],
            },
            generator: GeneratorType::Wilson,
            solver: SolverType::AStar,
            seed: 1,
        }
    }

    /// The `text` field of a serialized record.
    fn text_of(result: &MazeResult, placement: QueryPlacement) -> String {
        let bytes = process_batch(std::slice::from_ref(result), placement);
        let line = String::from_utf8(bytes).unwrap();
        let value: serde_json::Value = serde_json::from_str(line.trim_end()).unwrap();
        value["text"].as_str().unwrap().to_string()
    }

    #[test]
    fn query_follows_the_walls_by_default() {
        assert_eq!(
            text_of(&result_3x3(&[(1, 0), (0, 2)]), QueryPlacement::default()),
            "wall 1 0 wall 0 2 query start 0 0 goal 2 2 \
             reasoning create 0 0 c0 c4 close 0 0 c0 c4 solution plan 0 0 plan 2 2 end"
        );
    }

    #[test]
    fn legacy_placement_puts_the_query_first() {
        assert_eq!(
            text_of(&result_3x3(&[(1, 0), (0, 2)]), QueryPlacement::First),
            "query start 0 0 goal 2 2 wall 1 0 wall 0 2 \
             reasoning create 0 0 c0 c4 close 0 0 c0 c4 solution plan 0 0 plan 2 2 end"
        );
    }

    #[test]
    fn placement_only_reorders_the_prompt() {
        // Same tokens, same trace, same wall order: only the query block moves.
        let result = result_3x3(&[(1, 0), (0, 2), (2, 1)]);
        let new = text_of(&result, QueryPlacement::AfterWalls);
        let legacy = text_of(&result, QueryPlacement::First);
        assert_ne!(new, legacy);
        let mut new_tokens: Vec<&str> = new.split(' ').collect();
        let mut legacy_tokens: Vec<&str> = legacy.split(' ').collect();
        assert_eq!(new_tokens.len(), legacy_tokens.len());
        new_tokens.sort_unstable();
        legacy_tokens.sort_unstable();
        assert_eq!(new_tokens, legacy_tokens);
    }

    #[test]
    fn wall_free_mazes_have_no_stray_separator() {
        // Nothing precedes the query block, so `text` must not open with a space.
        for placement in [QueryPlacement::AfterWalls, QueryPlacement::First] {
            let text = text_of(&result_3x3(&[]), placement);
            assert!(
                text.starts_with("query start 0 0 goal 2 2 reasoning"),
                "{text}"
            );
            assert!(!text.contains("  "), "{text}");
        }
    }

    #[test]
    fn the_shared_helper_writes_the_records_own_trace_and_plan() {
        // `--puzzle-file` splices this fragment onto a record's original prompt, so it has to
        // be exactly what the record writer would have emitted there.
        let result = result_3x3(&[(1, 0)]);
        let text = text_of(&result, QueryPlacement::AfterWalls);
        let mut fragment = Vec::new();
        let mut buffer = String::new();
        write_trace_and_plan(&mut fragment, &result.solution, &mut buffer).unwrap();
        let fragment = String::from_utf8(fragment).unwrap();
        assert!(fragment.starts_with(" reasoning") && fragment.ends_with(" end"));
        assert_eq!(&text[text.find(" reasoning").unwrap()..], fragment);
    }

    #[test]
    fn batches_larger_than_a_chunk_serialize_identically() {
        // The parallel path must not disagree with the serial one about placement.
        let results: Vec<MazeResult> = (0..SERIALIZE_CHUNK * 2 + 1)
            .map(|_| result_3x3(&[(1, 0)]))
            .collect();
        let parallel = process_batch(&results, QueryPlacement::AfterWalls);
        let serial: Vec<u8> = results
            .iter()
            .flat_map(|r| process_batch(std::slice::from_ref(r), QueryPlacement::AfterWalls))
            .collect();
        assert_eq!(parallel, serial);
    }
}
