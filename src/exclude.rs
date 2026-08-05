//! `--exclude`: read the partition files of an existing dataset directory and hand back each
//! line's wall-layout key, so the seen-set can be pre-seeded with layouts that must never be
//! emitted again — keeping the new dataset disjoint from the old one.
//!
//! Matching is by WALLS ONLY (see `dedup`): a layout present in an excluded file is skipped
//! whatever its start/goal, so the new dataset shares no search graph with the old one — not
//! just no identical instance.
//!
//! Parsing is by hand (the writer emits a fixed key order and an escape-free `text`), which
//! keeps a multi-million-line scan cheap: only the prompt half of each line is decoded, the
//! trace is never touched.

use crate::dedup::{key_from_parts, LayoutKey};
use crate::types::Maze;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Read buffer for the line scan.
const READ_BUFFER: usize = 8 * 1024 * 1024;

/// Keys of the emitted JSON record.
const TEXT_KEY: &str = "\"text\":\"";
const ROWS_KEY: &str = "\"rows\":";
const COLS_KEY: &str = "\"cols\":";

/// Separator between the prompt (walls) and the search trace.
const TRACE_MARKER: &str = " reasoning";

/// Tallies from one scan, for the stderr summary.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct ScanStats {
    /// Lines parsed across all files.
    pub lines: u64,
    /// Lines whose grid dimensions differ from the run's (they can never collide, so they are
    /// dropped rather than keyed).
    pub skipped_dims: u64,
}

/// Existing partition files of a dataset directory, in fixed eval/test/train order.
///
/// Empty if the directory holds none of them (a mistyped `--exclude` path excludes nothing,
/// which the caller reports).
pub fn partition_files_in(dir: &str) -> Vec<String> {
    crate::cli::PartitionKind::ORDER
        .iter()
        .map(|k| Path::new(dir).join(format!("{}.jsonl", k.stem())))
        .filter(|p| p.is_file())
        .map(|p| p.to_string_lossy().into_owned())
        .collect()
}

/// All-floor packed grid for `rows` x `cols`, matching `Maze`'s bit packing.
///
/// Cloned per line and punched with the record's walls — far cheaper than setting every cell.
fn floor_template(rows: usize, cols: usize) -> Vec<u8> {
    let mut maze = Maze::new(rows, cols);
    for y in 0..rows {
        for x in 0..cols {
            maze.set_cell(x, y, true);
        }
    }
    maze.grid
}

/// Clear one cell (mark it a wall) in a packed grid.
fn clear_cell(grid: &mut [u8], cols_bytes: usize, x: usize, y: usize) {
    grid[y * cols_bytes + (x / 8)] &= !(1 << (x % 8));
}

/// Read the integer value of `key` from the record's tail.
fn parse_field(tail: &str, key: &str) -> Result<usize, String> {
    let at = tail.find(key).ok_or_else(|| format!("no {key} key"))?;
    tail[at + key.len()..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .map_err(|_| format!("unparsable {key} value"))
}

/// Layout key of one JSONL line, or `None` when the record's grid shape differs from the run's
/// (a different shape can never collide, so it is not worth keying).
fn parse_line(
    line: &str,
    rows: usize,
    cols: usize,
    template: &[u8],
) -> Result<Option<LayoutKey>, String> {
    let at = line
        .find(TEXT_KEY)
        .ok_or_else(|| format!("no {TEXT_KEY} key"))?;
    let rest = &line[at + TEXT_KEY.len()..];
    let end = rest
        .find('"')
        .ok_or_else(|| "unterminated \"text\" value".to_string())?;
    let (text, tail) = rest.split_at(end);

    if parse_field(tail, ROWS_KEY)? != rows || parse_field(tail, COLS_KEY)? != cols {
        return Ok(None);
    }

    // Walls live in the prompt, whichever side of it the query block sits on (see
    // `QueryPlacement`); the trace is the bulk of the line, so only the prompt is scanned.
    let head = match text.split_once(TRACE_MARKER) {
        Some((head, _)) => head,
        None => return Err("no ` reasoning` marker in \"text\"".into()),
    };

    let cols_bytes = cols.div_ceil(8);
    let mut grid = template.to_vec();
    let mut tokens = head.split_whitespace();
    while let Some(token) = tokens.next() {
        if token != "wall" {
            continue;
        }
        let mut coord = |axis: &str, bound: usize| -> Result<usize, String> {
            let raw = tokens
                .next()
                .ok_or_else(|| format!("truncated wall {axis} coordinate"))?;
            match raw.parse::<usize>() {
                Ok(v) if v < bound => Ok(v),
                Ok(v) => Err(format!("wall {axis}={v} outside the {rows}x{cols} grid")),
                Err(_) => Err(format!("unparsable wall {axis} coordinate '{raw}'")),
            }
        };
        let x = coord("x", cols)?;
        let y = coord("y", rows)?;
        clear_cell(&mut grid, cols_bytes, x, y);
    }

    Ok(Some(key_from_parts(rows, cols, &grid)))
}

/// Scan every path, handing each same-shape line's layout key to `sink`.
///
/// Lines whose grid shape differs are counted and dropped (they cannot collide); anything
/// unparsable is a hard error — a silently ignored line would leave the new dataset
/// overlapping the old.
pub fn scan_files<F>(
    paths: &[String],
    rows: usize,
    cols: usize,
    mut sink: F,
) -> Result<ScanStats, String>
where
    F: FnMut(LayoutKey),
{
    let template = floor_template(rows, cols);
    let mut stats = ScanStats::default();
    for path in paths {
        let file = File::open(path).map_err(|e| format!("--exclude {path}: {e}"))?;
        let mut reader = BufReader::with_capacity(READ_BUFFER, file);
        let mut line = String::new();
        let mut lineno = 0u64;
        loop {
            line.clear();
            let read = reader
                .read_line(&mut line)
                .map_err(|e| format!("--exclude {path}: {e}"))?;
            if read == 0 {
                break;
            }
            lineno += 1;
            let trimmed = line.trim_end();
            if trimmed.is_empty() {
                continue;
            }
            stats.lines += 1;
            match parse_line(trimmed, rows, cols, &template)
                .map_err(|e| format!("--exclude {path}:{lineno}: {e}"))?
            {
                Some(key) => sink(key),
                None => stats.skipped_dims += 1,
            }
        }
    }
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dedup::layout_key;
    use crate::serializer::{process_batch, QueryPlacement};
    use crate::types::{MazeResult, Solution};
    use crate::{GeneratorType, SolverType};
    use std::io::Write;

    /// A 3x3 record with walls at (1,0) and (0,2), as the serializer emits it by default
    /// (query block after the wall list).
    const LINE: &str = r#"{"idx":7,"text":"wall 1 0 wall 0 2 query start 0 0 goal 2 2 reasoning create 0 0 c0 c4 close 0 0 c0 c4 solution plan 0 0 plan 2 2 end","generator":"wilson","solver":"astar","seed":1,"rows":3,"cols":3}"#;

    /// The same record in the legacy `--legacy-query-first` layout.
    const LEGACY_LINE: &str = r#"{"idx":7,"text":"query start 0 0 goal 2 2 wall 1 0 wall 0 2 reasoning create 0 0 c0 c4 close 0 0 c0 c4 solution plan 0 0 plan 2 2 end","generator":"wilson","solver":"astar","seed":1,"rows":3,"cols":3}"#;

    fn maze_3x3(walls: &[(usize, usize)]) -> Maze {
        let mut maze = Maze::new(3, 3);
        for y in 0..3 {
            for x in 0..3 {
                maze.set_cell(x, y, true);
            }
        }
        for &(x, y) in walls {
            maze.set_cell(x, y, false);
        }
        maze
    }

    #[test]
    fn parsed_key_matches_the_generated_layout() {
        // The whole point: a key read back from disk must equal the key of the maze that
        // produced the line, so exclusion actually excludes.
        let template = floor_template(3, 3);
        let parsed = parse_line(LINE, 3, 3, &template).unwrap().unwrap();
        assert_eq!(parsed, layout_key(&maze_3x3(&[(1, 0), (0, 2)])));
        // A different wall set must not match.
        assert_ne!(parsed, layout_key(&maze_3x3(&[(1, 0)])));
    }

    #[test]
    fn wall_free_records_parse() {
        let line = r#"{"idx":0,"text":"query start 0 0 goal 2 2 reasoning close 0 0 c0 c4 solution plan 0 0 end","generator":"dfs","solver":"astar","seed":0,"rows":3,"cols":3}"#;
        let template = floor_template(3, 3);
        let parsed = parse_line(line, 3, 3, &template).unwrap().unwrap();
        assert_eq!(parsed, layout_key(&maze_3x3(&[])));
    }

    #[test]
    fn other_grid_shapes_are_skipped_not_keyed() {
        let template = floor_template(4, 4);
        assert_eq!(parse_line(LINE, 4, 4, &template).unwrap(), None);
    }

    #[test]
    fn malformed_lines_are_errors() {
        let template = floor_template(3, 3);
        // Not a line this tool wrote.
        assert!(parse_line(r#"{"foo":1}"#, 3, 3, &template).is_err());
        // Coordinate outside the grid.
        let oob = LINE.replace("wall 1 0", "wall 9 0");
        assert!(parse_line(&oob, 3, 3, &template)
            .unwrap_err()
            .contains("outside"));
        // Truncated coordinate pair.
        let truncated = LINE.replace("wall 0 2 query", "wall 0 query");
        assert!(parse_line(&truncated, 3, 3, &template).is_err());
    }

    #[test]
    fn both_query_placements_key_the_same_layout() {
        // Exclusion must work against datasets written in either text layout.
        let template = floor_template(3, 3);
        let expected = layout_key(&maze_3x3(&[(1, 0), (0, 2)]));
        assert_eq!(
            parse_line(LINE, 3, 3, &template).unwrap().unwrap(),
            expected
        );
        assert_eq!(
            parse_line(LEGACY_LINE, 3, 3, &template).unwrap().unwrap(),
            expected
        );
    }

    #[test]
    fn serializer_output_round_trips() {
        // Guards the hand-written parser against a change in what the writer emits.
        let walls = [(1, 0), (0, 2)];
        let mut maze = maze_3x3(&walls);
        maze.start = (0, 0);
        maze.goal = (2, 2);
        let expected = layout_key(&maze);
        let result = MazeResult {
            instance_id: 0,
            maze,
            solution: Solution {
                path: vec![(0, 0)],
                reasoning: Vec::new(),
            },
            generator: GeneratorType::Wilson,
            solver: SolverType::AStar,
            seed: 0,
        };
        let template = floor_template(3, 3);
        for placement in [QueryPlacement::AfterWalls, QueryPlacement::First] {
            let bytes = process_batch(std::slice::from_ref(&result), placement);
            let line = String::from_utf8(bytes).unwrap();
            assert_eq!(
                parse_line(line.trim_end(), 3, 3, &template)
                    .unwrap()
                    .unwrap(),
                expected,
                "placement {placement:?}"
            );
        }
    }

    fn write_tmp(tag: &str, body: &str) -> String {
        let path =
            std::env::temp_dir().join(format!("maze_exclude_{}_{}.jsonl", tag, std::process::id()));
        let mut f = File::create(&path).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        path.to_string_lossy().into_owned()
    }

    #[test]
    fn scan_collects_keys_and_skips_other_shapes() {
        let other_shape = LINE.replace(r#""rows":3,"cols":3"#, r#""rows":5,"cols":5"#);
        let body = format!("{LINE}\n{other_shape}\n\n{LINE}\n");
        let path = write_tmp("scan", &body);

        let mut keys: Vec<LayoutKey> = Vec::new();
        let stats = scan_files(std::slice::from_ref(&path), 3, 3, |k| keys.push(k)).unwrap();

        assert_eq!(
            stats,
            ScanStats {
                lines: 3,
                skipped_dims: 1
            }
        );
        assert_eq!(
            keys,
            vec![
                layout_key(&maze_3x3(&[(1, 0), (0, 2)])),
                layout_key(&maze_3x3(&[(1, 0), (0, 2)]))
            ]
        );

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn scan_reports_the_offending_file_and_line() {
        let body = format!("{LINE}\nnot json\n");
        let path = write_tmp("badline", &body);
        let err = scan_files(std::slice::from_ref(&path), 3, 3, |_| {}).unwrap_err();
        assert!(err.contains(&format!("{path}:2")), "unhelpful error: {err}");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_file_is_an_error() {
        let err = scan_files(&["/nonexistent/nope.jsonl".into()], 3, 3, |_| {}).unwrap_err();
        assert!(err.contains("--exclude"), "unhelpful error: {err}");
    }

    #[test]
    fn partition_files_lists_present_ones_in_order() {
        let dir = std::env::temp_dir().join(format!("maze_pf_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Only eval and train present (no test).
        for stem in ["eval", "train"] {
            File::create(dir.join(format!("{stem}.jsonl"))).unwrap();
        }
        let got = partition_files_in(&dir.to_string_lossy());
        let names: Vec<String> = got
            .iter()
            .map(|p| {
                Path::new(p)
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        assert_eq!(names, vec!["eval.jsonl", "train.jsonl"]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
