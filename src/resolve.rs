//! `--puzzle-file`: re-solve the mazes in an existing JSONL and write `<stem>_resolved.jsonl`
//! beside it.
//!
//! The contract is that nothing but the trace and the plan changes. Every JSON key is copied
//! VERBATIM — including keys this tool knows nothing about, and their order — and inside `text`
//! everything before ` reasoning` is kept byte for byte. Only the search trace and the plan are
//! replaced, by this crate's A* run over the maze the record itself describes.
//!
//! That is done by splicing the raw line rather than decoding it: `serde_json` here is built
//! without `preserve_order`, so a `Value` round-trip would silently re-sort every record's keys.
//! Head and tail are the input's own bytes and the middle is escape-free ASCII from our own
//! writer, so the result is valid JSON by construction.
//!
//! The motivating input is a `test_looped.jsonl` (see `make_looped_maze_test.py` in the parent
//! repo): a real maze with one wall deleted and the pre-deletion trace still attached. Those
//! records carry an extra `removed` key, which must survive untouched.

use crate::cli::ResolveOpts;
use crate::jsonl::{locate_text, parse_field, COLS_KEY, READ_BUFFER, ROWS_KEY, TRACE_MARKER};
use crate::serializer::write_trace_and_plan;
use crate::solvers;
use crate::types::Maze;
use crate::SolverType;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Lines read before each parallel re-solve pass.
const CHUNK: usize = 4096;

/// Lines handed to one rayon task; mirrors `serializer::SERIALIZE_CHUNK`.
const SUB_CHUNK: usize = 32;

/// Appended to the input's file stem to name the output.
const RESOLVED_SUFFIX: &str = "_resolved";

/// Output path for a puzzle file: `<dir>/<stem>_resolved.jsonl`.
///
/// The name can never collide with the input, so a resolve run cannot read and write the same
/// file (`x_resolved.jsonl` would resolve to `x_resolved_resolved.jsonl`).
pub fn resolved_path(input: &Path) -> Result<PathBuf, String> {
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("--puzzle-file {}: not a file name", input.display()))?;
    Ok(input.with_file_name(format!("{stem}{RESOLVED_SUFFIX}.jsonl")))
}

/// How one record's re-solve differs from what was stored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct RecordDelta {
    /// The fresh trace and plan are byte-identical to the stored ones.
    unchanged: bool,
    /// New minus old, in plan steps.
    plan: i64,
    /// New minus old, in trace events.
    trace: i64,
}

/// What a whole file's re-solve changed.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ResolveStats {
    pub records: u64,
    pub blank_lines: u64,
    pub unchanged: u64,
    pub plan_shorter: u64,
    pub plan_longer: u64,
    pub trace_shorter: u64,
    pub trace_longer: u64,
    /// Summed new-minus-old over the records that changed.
    pub plan_delta_sum: i64,
    pub trace_delta_sum: i64,
}

impl ResolveStats {
    fn add(&mut self, delta: Option<RecordDelta>) {
        let Some(delta) = delta else {
            self.blank_lines += 1;
            return;
        };
        self.records += 1;
        if delta.unchanged {
            self.unchanged += 1;
            return;
        }
        self.plan_delta_sum += delta.plan;
        self.trace_delta_sum += delta.trace;
        match delta.plan.signum() {
            -1 => self.plan_shorter += 1,
            1 => self.plan_longer += 1,
            _ => {}
        }
        match delta.trace.signum() {
            -1 => self.trace_shorter += 1,
            1 => self.trace_longer += 1,
            _ => {}
        }
    }

    fn changed(&self) -> u64 {
        self.records - self.unchanged
    }
}

/// One `wall`/`start`/`goal` segment of the maze spec; each takes an (x, y) pair.
enum Segment {
    Wall,
    Start,
    Goal,
}

/// Rebuild the maze from the prompt half of `text` (everything before ` reasoning`).
///
/// The tokens are read order-independently, so both query placements parse: the query block may
/// precede or follow the wall list.
///
/// Unlike the `--exclude` scan, an unrecognised token is an error rather than something to skip:
/// this mode's entire output is derived from this parse, so a token we do not understand means
/// we may be misreading the maze.
fn parse_prompt(prompt: &str, rows: usize, cols: usize) -> Result<Maze, String> {
    let mut maze = Maze::all_floor(rows, cols);
    let mut start: Option<(usize, usize)> = None;
    let mut goal: Option<(usize, usize)> = None;

    let mut tokens = prompt.split_whitespace();
    while let Some(token) = tokens.next() {
        let segment = match token {
            // A bare keyword: the coordinates that follow belong to `start` and `goal`.
            "query" => continue,
            "wall" => Segment::Wall,
            "start" => Segment::Start,
            "goal" => Segment::Goal,
            other => return Err(format!("unexpected token '{other}' in the maze spec")),
        };
        let mut coord = |axis: &str, bound: usize| -> Result<usize, String> {
            let raw = tokens
                .next()
                .ok_or_else(|| format!("truncated {token} {axis} coordinate"))?;
            match raw.parse::<usize>() {
                Ok(v) if v < bound => Ok(v),
                Ok(v) => Err(format!("{token} {axis}={v} outside the {rows}x{cols} grid")),
                Err(_) => Err(format!("unparsable {token} {axis} coordinate '{raw}'")),
            }
        };
        let x = coord("x", cols)?;
        let y = coord("y", rows)?;
        match segment {
            Segment::Wall => maze.set_cell(x, y, false),
            Segment::Start => match start {
                Some(_) => return Err("duplicate start in the maze spec".into()),
                None => start = Some((x, y)),
            },
            Segment::Goal => match goal {
                Some(_) => return Err("duplicate goal in the maze spec".into()),
                None => goal = Some((x, y)),
            },
        }
    }

    let start = start.ok_or("no start in the maze spec")?;
    let goal = goal.ok_or("no goal in the maze spec")?;
    // Checked after every wall is punched, so a start also listed as a wall is caught. A*
    // indexes its score arrays with the start unguarded, so a walled or out-of-grid start is a
    // panic, not a bad trace.
    if !maze.get_cell(start.0, start.1) {
        return Err(format!("start {start:?} is a wall"));
    }
    if !maze.get_cell(goal.0, goal.1) {
        return Err(format!("goal {goal:?} is a wall"));
    }
    // A* reconstructs no path for this, which would be indistinguishable from unsolvable.
    if start == goal {
        return Err(format!("start and goal are the same cell {start:?}"));
    }
    maze.start = start;
    maze.goal = goal;
    Ok(maze)
}

/// Grid dimensions of a record.
///
/// `text` is escape-free, so it can hold no `"` and these keys can only match outside it — the
/// whole line is searchable whatever the key order.
fn dims(raw: &str) -> Result<(usize, usize), String> {
    let dim = |key| {
        parse_field(raw, key)
            .map_err(|e| format!("{e}: --puzzle-file needs the record's grid dimensions"))
    };
    let (rows, cols) = (dim(ROWS_KEY)?, dim(COLS_KEY)?);
    if rows == 0 || cols == 0 {
        return Err(format!("degenerate {rows}x{cols} grid"));
    }
    // A*'s g and f scores are u16 and wrap silently in release builds.
    if rows.saturating_mul(cols).saturating_add(rows + cols) > u16::MAX as usize {
        return Err(format!("{rows}x{cols} grid is too large: A*'s costs are u16"));
    }
    Ok((rows, cols))
}

/// Re-solve one record, appending the spliced line to `out`.
///
/// `raw` is the line as read, newline included, and that newline travels through with the rest
/// of the record. Returns `None` for a blank line, which is passed through untouched.
fn resolve_line(
    raw: &str,
    solver: SolverType,
    out: &mut Vec<u8>,
    buffer: &mut String,
) -> Result<Option<RecordDelta>, String> {
    if raw.trim().is_empty() {
        out.extend_from_slice(raw.as_bytes());
        return Ok(None);
    }
    let body = raw.trim_end();
    if !body.starts_with('{') || !body.ends_with('}') {
        return Err("not a JSON object".into());
    }

    let span = locate_text(raw)?;
    let text = &raw[span.value_start..span.close_quote];
    // What makes the first `"` the true end of the value, and the splice valid JSON.
    if text.contains('\\') {
        return Err("escape sequence in \"text\"; --puzzle-file needs escape-free records".into());
    }
    // A record that lost its closing quote would otherwise take the *next* key's quote as the
    // terminator and splice into something that is no longer JSON. After a string value the
    // next character can only be `,` or `}`.
    let after = raw[span.close_quote + 1..].trim_start();
    if !(after.starts_with(',') || after.starts_with('}')) {
        return Err("unterminated \"text\" value (no ',' or '}' after it)".into());
    }
    let marker = text
        .find(TRACE_MARKER)
        .ok_or_else(|| format!("no `{TRACE_MARKER}` marker in \"text\""))?;

    let (rows, cols) = dims(raw)?;
    let maze = parse_prompt(&text[..marker], rows, cols)?;
    let solution = solvers::solve_maze(solver, &maze);
    if solution.path.is_empty() {
        return Err(format!(
            "no path from start {:?} to goal {:?}",
            maze.start, maze.goal
        ));
    }

    // The prompt, verbatim; then the fresh trace and plan; then every remaining key, verbatim.
    let head_end = span.value_start + marker;
    let stored = &raw[head_end..span.close_quote];
    out.extend_from_slice(&raw.as_bytes()[..head_end]);
    let fresh_at = out.len();
    write_trace_and_plan(out, &solution, buffer).expect("writing to a Vec cannot fail");
    let unchanged = out[fresh_at..] == *stored.as_bytes();
    out.extend_from_slice(&raw.as_bytes()[span.close_quote..]);

    // Token counts are only needed for the summary, so only pay for them when something moved.
    let delta = match unchanged {
        true => RecordDelta {
            unchanged: true,
            ..RecordDelta::default()
        },
        false => {
            let count = |kinds: &[&str]| {
                stored
                    .split_whitespace()
                    .filter(|t| kinds.contains(t))
                    .count() as i64
            };
            RecordDelta {
                unchanged: false,
                plan: solution.path.len() as i64 - count(&["plan"]),
                trace: solution.reasoning.len() as i64 - count(&["create", "close"]),
            }
        }
    };
    Ok(Some(delta))
}

/// Re-solve a chunk of lines in parallel, appending the spliced output in input order.
///
/// On failure the reported line is the earliest failing one, at any thread count: the sub-chunk
/// results are walked in index order rather than short-circuited by rayon, whose
/// `collect::<Result<..>>` gives no lowest-index guarantee.
fn resolve_chunk(
    lines: &[String],
    first_lineno: u64,
    solver: SolverType,
    out: &mut Vec<u8>,
    stats: &mut ResolveStats,
) -> Result<(), (u64, String)> {
    type SubResult = Result<(Vec<u8>, Vec<Option<RecordDelta>>), (u64, String)>;
    let results: Vec<SubResult> = lines
        .par_chunks(SUB_CHUNK)
        .enumerate()
        .map(|(i, sub)| {
            let base = first_lineno + (i * SUB_CHUNK) as u64;
            let mut bytes = Vec::with_capacity(sub.len() * 8192);
            let mut buffer = String::with_capacity(64);
            let mut deltas = Vec::with_capacity(sub.len());
            for (j, raw) in sub.iter().enumerate() {
                match resolve_line(raw, solver, &mut bytes, &mut buffer) {
                    Ok(delta) => deltas.push(delta),
                    Err(e) => return Err((base + j as u64, e)),
                }
            }
            Ok((bytes, deltas))
        })
        .collect();

    for result in results {
        let (bytes, deltas) = result?;
        out.extend_from_slice(&bytes);
        for delta in deltas {
            stats.add(delta);
        }
    }
    Ok(())
}

/// A partially written output file, removed on drop unless committed.
///
/// An unsolvable record aborts the run, so the output is built under a temporary name and
/// renamed into place only on success: the final path never holds a partial file, not even if
/// the process dies mid-run.
struct TempOut {
    path: PathBuf,
    target: PathBuf,
    committed: bool,
}

impl TempOut {
    /// Create `<target>.tmp.<pid>` beside the target. The pid keeps concurrent runs apart, and
    /// `create_new` makes a leftover sibling a loud error rather than a silent truncation.
    fn create(target: &Path) -> Result<(Self, File), String> {
        let name = target
            .file_name()
            .ok_or_else(|| format!("{}: not a file name", target.display()))?;
        let mut tmp_name = name.to_os_string();
        tmp_name.push(format!(".tmp.{}", std::process::id()));
        let path = target.with_file_name(tmp_name);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        Ok((
            Self {
                path,
                target: target.to_path_buf(),
                committed: false,
            },
            file,
        ))
    }

    /// Move the finished file into place.
    fn commit(mut self) -> Result<(), String> {
        fs::rename(&self.path, &self.target)
            .map_err(|e| format!("{} -> {}: {e}", self.path.display(), self.target.display()))?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for TempOut {
    fn drop(&mut self) {
        if !self.committed {
            // Best effort: an aborted run must not leave a half-written file behind.
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Stream a whole file: chunked read, parallel re-solve, ordered write.
fn resolve_stream<R: BufRead, W: Write>(
    reader: &mut R,
    writer: &mut W,
    input: &Path,
    solver: SolverType,
    progress: &ProgressBar,
) -> Result<ResolveStats, String> {
    let mut stats = ResolveStats::default();
    let mut lines: Vec<String> = Vec::with_capacity(CHUNK);
    let mut out: Vec<u8> = Vec::new();
    let mut lineno = 1u64;
    let mut consumed = 0u64;

    loop {
        // Fill the chunk, reusing the line allocations across chunks.
        let mut n = 0;
        while n < CHUNK {
            if lines.len() == n {
                lines.push(String::new());
            } else {
                lines[n].clear();
            }
            let read = reader
                .read_line(&mut lines[n])
                .map_err(|e| format!("--puzzle-file {}: {e}", input.display()))?;
            if read == 0 {
                break;
            }
            consumed += read as u64;
            n += 1;
        }
        if n == 0 {
            break;
        }
        out.clear();
        resolve_chunk(&lines[..n], lineno, solver, &mut out, &mut stats)
            .map_err(|(l, e)| format!("--puzzle-file {}:{l}: {e}", input.display()))?;
        writer
            .write_all(&out)
            .map_err(|e| format!("{}: {e}", input.display()))?;
        lineno += n as u64;
        progress.set_position(consumed);
    }
    Ok(stats)
}

/// Report what the re-solve changed.
fn report(stats: &ResolveStats, output: &Path, elapsed: Duration) {
    println!(
        "\nResolved {} records into {} in {:.2}s",
        stats.records,
        output.display(),
        elapsed.as_secs_f64()
    );
    if stats.blank_lines > 0 {
        println!("  {} blank line(s) passed through", stats.blank_lines);
    }
    println!(
        "  {}/{} unchanged (the stored trace and plan already match this A*)",
        stats.unchanged, stats.records
    );
    let changed = stats.changed();
    if changed == 0 {
        eprintln!(
            "Warning: nothing changed — every stored trace already matched this A*. \
             Is this the file you meant?"
        );
        return;
    }
    let mean = |sum: i64| sum as f64 / changed as f64;
    println!(
        "  plan:  shorter {}, longer {} (mean {:+.1} steps over the {} changed)",
        stats.plan_shorter,
        stats.plan_longer,
        mean(stats.plan_delta_sum),
        changed
    );
    println!(
        "  trace: shorter {}, longer {} (mean {:+.1} events over the {} changed)",
        stats.trace_shorter,
        stats.trace_longer,
        mean(stats.trace_delta_sum),
        changed
    );
    if stats.plan_longer > 0 {
        eprintln!(
            "Warning: {} record(s) got a LONGER optimal plan; \
             the stored plan was not optimal for the stored maze.",
            stats.plan_longer
        );
    }
}

/// Re-solve one puzzle file into `output`.
fn resolve_file(
    input: &Path,
    output: &Path,
    solver: SolverType,
) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    println!("Re-solving {} -> {} ...", input.display(), output.display());

    let file = File::open(input).map_err(|e| format!("--puzzle-file {}: {e}", input.display()))?;
    let total = file.metadata().map(|m| m.len()).unwrap_or(0);
    let mut reader = BufReader::with_capacity(READ_BUFFER, file);

    // Measured in bytes, so there is no line-counting pre-pass over a multi-GB file.
    let progress = ProgressBar::new(total);
    progress.set_style(
        ProgressStyle::default_bar()
            .template(
                "[{elapsed_precise}] {bar:40.green/red} {bytes}/{total_bytes} ({bytes_per_sec})",
            )
            .expect("valid progress template")
            .progress_chars("##-"),
    );

    let (temp, out_file) = TempOut::create(output)?;
    let mut writer = BufWriter::with_capacity(READ_BUFFER, out_file);
    let stats = match resolve_stream(&mut reader, &mut writer, input, solver, &progress) {
        Ok(stats) => stats,
        Err(e) => {
            // Leave the bar where it stopped rather than let it redraw over the error.
            progress.abandon();
            return Err(e.into());
        }
    };
    writer.flush()?;
    drop(writer);
    temp.commit()?;
    progress.finish_and_clear();

    report(&stats, output, started.elapsed());
    Ok(())
}

/// Re-solve every requested puzzle file.
pub fn run(opts: &ResolveOpts) -> Result<(), Box<dyn std::error::Error>> {
    // Every clobber check first, so a run that cannot finish writes nothing at all.
    let outputs = opts
        .inputs
        .iter()
        .map(|input| resolved_path(input))
        .collect::<Result<Vec<_>, _>>()?;
    for output in &outputs {
        if output.exists() {
            return Err(format!(
                "refusing to overwrite existing {} (remove it or move the input)",
                output.display()
            )
            .into());
        }
    }
    for (input, output) in opts.inputs.iter().zip(&outputs) {
        resolve_file(input, output, opts.solver)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serializer::{process_batch, QueryPlacement};
    use crate::types::MazeResult;
    use crate::GeneratorType;

    /// Build a maze from an ASCII map: `#` wall, `S` start, `G` goal.
    fn maze_from(map: &[&str]) -> Maze {
        let mut maze = Maze::all_floor(map.len(), map[0].len());
        for (y, line) in map.iter().enumerate() {
            for (x, ch) in line.chars().enumerate() {
                match ch {
                    '#' => maze.set_cell(x, y, false),
                    'S' => maze.start = (x, y),
                    'G' => maze.goal = (x, y),
                    _ => {}
                }
            }
        }
        maze
    }

    /// A record for `map`, as the generator would have written it.
    fn record_line(map: &[&str], placement: QueryPlacement) -> String {
        let maze = maze_from(map);
        let solution = solvers::solve_maze(SolverType::AStar, &maze);
        let result = MazeResult {
            instance_id: 7,
            maze,
            solution,
            generator: GeneratorType::Wilson,
            solver: SolverType::AStar,
            seed: 1,
        };
        String::from_utf8(process_batch(std::slice::from_ref(&result), placement)).unwrap()
    }

    /// Re-solve one line, returning the spliced output and the delta.
    fn resolve(raw: &str) -> Result<(String, Option<RecordDelta>), String> {
        let mut out = Vec::new();
        let mut buffer = String::new();
        let delta = resolve_line(raw, SolverType::AStar, &mut out, &mut buffer)?;
        Ok((String::from_utf8(out).unwrap(), delta))
    }

    #[test]
    fn resolving_an_unmodified_record_reproduces_it_byte_for_byte() {
        // The load-bearing test: it pins the splice offsets, the trace/plan helper, the prompt
        // parser and A*'s determinism at once. A* is deterministic, so re-solving a record this
        // crate wrote has to give that record back, newline included.
        let map = ["S..#.", ".#...", ".#.#.", "...#G"];
        for placement in [QueryPlacement::AfterWalls, QueryPlacement::First] {
            let line = record_line(&map, placement);
            let (out, delta) = resolve(&line).unwrap();
            assert_eq!(out, line, "placement {placement:?}");
            assert!(delta.unwrap().unchanged);
        }
    }

    #[test]
    fn unknown_keys_and_key_order_survive() {
        // Keys this tool knows nothing about, in an order it never writes, with `rows`/`cols`
        // ahead of `text` — all of it has to come back untouched.
        let line = concat!(
            r#"{"rows":3,"cols":3,"removed":[5,10],"zz":{"a":1},"#,
            r#""text":"query start 0 0 goal 2 0 reasoning close 9 9 c9 c9 solution plan 9 9 end","#,
            r#""idx":7}"#,
            "\n"
        );
        let (out, _) = resolve(line).unwrap();
        assert_eq!(
            out,
            concat!(
                r#"{"rows":3,"cols":3,"removed":[5,10],"zz":{"a":1},"#,
                r#""text":"query start 0 0 goal 2 0 reasoning close 0 0 c0 c2 create 1 0 c1 c1 "#,
                r#"create 0 1 c1 c3 close 1 0 c1 c1 create 2 0 c2 c0 create 1 1 c2 c2 "#,
                r#"close 2 0 c2 c0 solution plan 0 0 plan 1 0 plan 2 0 end","#,
                r#""idx":7}"#,
                "\n"
            )
        );
    }

    #[test]
    fn a_looped_record_gets_a_fresh_trace_and_plan() {
        // The motivating case: a wall has been deleted from `text` (opening a shortcut) but the
        // trace and plan still describe the route from before the deletion.
        let line = concat!(
            r#"{"idx":3,"text":"query start 0 0 goal 2 0 wall 1 2 reasoning "#,
            r#"close 0 0 c0 c2 create 0 1 c1 c3 close 0 1 c1 c3 create 0 2 c2 c4 "#,
            r#"solution plan 0 0 plan 0 1 plan 0 2 plan 1 2 plan 2 2 plan 2 1 plan 2 0 end","#,
            r#""rows":3,"cols":3,"removed":[1,0]}"#,
            "\n"
        );
        let (out, delta) = resolve(line).unwrap();
        let delta = delta.unwrap();
        assert!(!delta.unchanged);
        // The direct route along the top row is open: the 7 stored steps become 3.
        assert!(
            out.contains("solution plan 0 0 plan 1 0 plan 2 0 end"),
            "{out}"
        );
        assert_eq!(delta.plan, 3 - 7);
        // Everything outside the trace and plan is untouched, `removed` included.
        assert!(out.starts_with(r#"{"idx":3,"text":"query start 0 0 goal 2 0 wall 1 2 reasoning"#));
        assert!(out.ends_with(concat!(r#""rows":3,"cols":3,"removed":[1,0]}"#, "\n")));
    }

    #[test]
    fn blank_lines_pass_through() {
        let (out, delta) = resolve("\n").unwrap();
        assert_eq!(out, "\n");
        assert_eq!(delta, None);
    }

    #[test]
    fn a_final_line_without_a_newline_stays_that_way() {
        let line = record_line(&["S.", ".G"], QueryPlacement::AfterWalls);
        let trimmed = line.trim_end();
        let (out, _) = resolve(trimmed).unwrap();
        assert_eq!(out, trimmed);
    }

    #[test]
    fn malformed_records_are_errors() {
        // A valid line, then the ways it can be broken; each error has to name its cause.
        let good = concat!(
            r#"{"idx":7,"text":"query start 0 0 goal 2 0 wall 1 1 reasoning "#,
            r#"close 0 0 c0 c2 solution plan 0 0 end","rows":3,"cols":3}"#,
            "\n"
        );
        assert!(resolve(good).is_ok());

        let cases: &[(String, &str)] = &[
            // Python's default json.dumps spacing: this tool only reads compact records.
            (good.replace(r#""text":""#, r#""text": ""#), "text"),
            (good.replace(" reasoning", " thinking"), "reasoning"),
            (good.replace(r#""rows":3,"#, ""), "grid dimensions"),
            (good.replace(r#""cols":3"#, r#""cols":0"#), "degenerate"),
            (good.replace(r#""rows":3"#, r#""rows":99999"#), "too large"),
            (good.replace("wall 1 1", "wall 9 1"), "outside"),
            (
                good.replace("wall 1 1 reasoning", "wall 1 reasoning"),
                "truncated",
            ),
            (good.replace("wall 1 1", "wall x 1"), "unparsable"),
            (good.replace("wall 1 1", "cell 1 1"), "unexpected token"),
            (
                good.replace("start 0 0", "start 0 0 start 1 0"),
                "duplicate",
            ),
            (good.replace("query start 0 0", "query"), "no start"),
            (good.replace("goal 2 0", ""), "no goal"),
            (good.replace("start 0 0", "start 1 1"), "is a wall"),
            (good.replace("goal 2 0", "goal 0 0"), "same cell"),
            // The goal walled off from the start.
            (
                good.replace("wall 1 1", "wall 1 0 wall 1 1 wall 1 2"),
                "no path",
            ),
            (good.replace('{', " "), "not a JSON object"),
            (
                good.replace(r#" end","rows""#, r#" end,"rows""#),
                "unterminated",
            ),
            (good.replace("close 0 0", "clo\\se 0 0"), "escape sequence"),
        ];
        for (line, expect) in cases {
            let err = resolve(line).unwrap_err();
            assert!(err.contains(expect), "expected {expect:?}, got {err:?}");
        }
    }

    #[test]
    fn the_first_failing_line_is_reported() {
        // Bad lines in different sub-chunks: the earliest has to win at any thread count.
        let good = record_line(&["S.", ".G"], QueryPlacement::AfterWalls);
        let mut lines: Vec<String> = (0..100).map(|_| good.clone()).collect();
        lines[2] = good.replace("query", "cell 0 0 query");
        lines[SUB_CHUNK + 5] = "not json\n".to_string();
        for _ in 0..20 {
            let mut out = Vec::new();
            let mut stats = ResolveStats::default();
            let (lineno, _) =
                resolve_chunk(&lines, 1, SolverType::AStar, &mut out, &mut stats).unwrap_err();
            assert_eq!(lineno, 3);
        }
    }

    /// A temp directory holding `test.jsonl` with `body`.
    fn write_input(tag: &str, body: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("maze_resolve_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let input = dir.join("test.jsonl");
        fs::write(&input, body).unwrap();
        input
    }

    fn opts(input: &Path) -> ResolveOpts {
        ResolveOpts {
            inputs: vec![input.to_path_buf()],
            solver: SolverType::AStar,
        }
    }

    #[test]
    fn output_path_is_derived_from_the_input() {
        assert_eq!(
            resolved_path(Path::new("/a/b/test_looped.jsonl")).unwrap(),
            Path::new("/a/b/test_looped_resolved.jsonl")
        );
        assert_eq!(
            resolved_path(Path::new("train.jsonl")).unwrap(),
            Path::new("train_resolved.jsonl")
        );
        // A trailing slash is not a file name; `validate_resolve` rejects directories first.
        assert!(resolved_path(Path::new("/")).is_err());
    }

    #[test]
    fn a_whole_file_resolves() {
        let good = record_line(&["S.#", "...", "#.G"], QueryPlacement::First);
        let input = write_input("file", &format!("{good}\n{good}"));
        run(&opts(&input)).unwrap();
        let out = fs::read_to_string(input.with_file_name("test_resolved.jsonl")).unwrap();
        // Two records and a passed-through blank line, all byte-identical (nothing was stale).
        assert_eq!(out, format!("{good}\n{good}"));
        let _ = fs::remove_dir_all(input.parent().unwrap());
    }

    #[test]
    fn aborting_leaves_no_output() {
        // An unsolvable record on the last line: the run fails and nothing survives, neither
        // the target nor the temporary it was being built under.
        let good = record_line(&["S.", ".G"], QueryPlacement::AfterWalls);
        let walled = good.replace("query start", "wall 1 0 wall 0 1 query start");
        let input = write_input("abort", &format!("{good}{walled}"));
        let err = run(&opts(&input)).unwrap_err().to_string();
        assert!(err.contains("no path") && err.contains(":2"), "{err}");

        let dir = input.parent().unwrap();
        let leftovers: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|n| n != "test.jsonl")
            .collect();
        assert!(leftovers.is_empty(), "left behind: {leftovers:?}");
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn existing_output_is_not_clobbered() {
        let good = record_line(&["S.", ".G"], QueryPlacement::AfterWalls);
        let input = write_input("clobber", &good);
        let output = input.with_file_name("test_resolved.jsonl");
        fs::write(&output, "precious\n").unwrap();
        let err = run(&opts(&input)).unwrap_err().to_string();
        assert!(err.contains("refusing to overwrite"), "{err}");
        assert_eq!(fs::read_to_string(&output).unwrap(), "precious\n");
        let _ = fs::remove_dir_all(input.parent().unwrap());
    }

    #[test]
    fn summary_counts_changed_records() {
        let good = record_line(&["S.", ".G"], QueryPlacement::AfterWalls);
        // A stale record: one plan step short of the truth.
        let stale = good.replace(" solution plan 0 0", " solution");
        let mut out = Vec::new();
        let mut stats = ResolveStats::default();
        let lines = vec![good, stale, "\n".to_string()];
        resolve_chunk(&lines, 1, SolverType::AStar, &mut out, &mut stats).unwrap();
        assert_eq!(stats.records, 2);
        assert_eq!(stats.blank_lines, 1);
        assert_eq!(stats.unchanged, 1);
        assert_eq!(stats.plan_longer, 1);
        assert_eq!(stats.plan_shorter, 0);
        assert_eq!(stats.plan_delta_sum, 1);
        assert_eq!(stats.trace_delta_sum, 0);
    }
}
