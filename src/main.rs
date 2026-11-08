mod balanced;
mod cli;
mod dedup;
mod exclude;
mod generators;
mod jsonl;
mod parameters;
mod prng;
mod resolve;
mod serializer;
mod solvers;
mod types;

use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Serialize;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::mpsc::{sync_channel, SyncSender};
use std::thread;
use std::time::Instant;

use crate::cli::{Args, BalancedOpts, Mode, Partition, PartitionKind, Validated};
use crate::dedup::{layout_key, query_key, SeenSet};
use crate::parameters::{print_all_params_help, print_param_help, GeneratorParams};
use crate::prng::create_instance_prng;
use crate::serializer::{process_batch, QueryPlacement};
use crate::types::{Maze, MazeResult};

#[derive(Clone, Copy, Debug, ValueEnum, Hash, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum GeneratorType {
    Dfs,
    Kruskal,
    Wilson,
    Searchformer,
    DrunkardsWalk,
}

#[derive(Clone, Copy, Debug, ValueEnum, Hash, PartialEq, Eq, Serialize)]
pub enum SolverType {
    #[value(name = "astar")]
    #[serde(rename = "astar")]
    AStar,
}

/// Batch size for serialization / sending to the writer.
const BATCH_SIZE: usize = 1000;

/// Candidate generation chunk: bounds wasted work past a partition's target while keeping
/// parallelism high. Leftovers carry over to the next partition, so nothing is thrown away
/// except at the end of the run.
const CHUNK: u64 = 8 * BATCH_SIZE as u64;

/// Instance id whose PRNG draws the `--fixed-maze` layout; the candidate stream counts up from 0
/// and never reaches it.
const FIXED_LAYOUT_ID: u64 = u64::MAX;

/// Shared maze-generation parameters; threaded through the candidate stream.
#[derive(Clone, Copy)]
struct GenSpec<'a> {
    seed: u64,
    generator: GeneratorType,
    solver: SolverType,
    rows: usize,
    cols: usize,
    params: &'a GeneratorParams,
    /// `--fixed-maze` layout shared by every instance; `None` = a fresh layout per instance.
    fixed: Option<&'a Maze>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Handle --list-params
    if let Some(maybe_generator) = args.list_params {
        match maybe_generator {
            Some(generator) => print_param_help(generator),
            None => print_all_params_help(),
        }
        return Ok(());
    }

    // Set thread pool size if specified; sizes both modes, and may only be set once.
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?;
    }

    let validated = match args.mode()? {
        Mode::Resolve(opts) => return resolve::run(&opts),
        Mode::Generate(validated) => *validated,
    };
    let generator_params = GeneratorParams::from_vec(args.params.clone())?;

    let start_time = Instant::now();
    println!(
        "Generating {} mazes into {} ...",
        validated.total_count(),
        validated.output_dir
    );

    let mut spec = GenSpec {
        seed: args.seed,
        generator: validated.generator,
        solver: validated.solver,
        rows: validated.rows,
        cols: validated.cols,
        params: &generator_params,
        fixed: None,
    };
    let fixed_layout = validated.fixed_maze.then(|| fixed_layout(spec));
    if let Some(layout) = &fixed_layout {
        let floors = generators::floors(layout).len() as u64;
        let queries = floors * (floors - 1);
        println!("Fixed maze: {floors} floor cells, {queries} distinct queries");
        if validated.total_count() > queries {
            return Err(format!(
                "--fixed-maze: requested {} mazes but the layout admits only {queries} \
                 distinct (start, goal) pairs; enlarge --rows/--cols or lower the counts",
                validated.total_count()
            )
            .into());
        }
    }
    spec.fixed = fixed_layout.as_ref();

    let (emitted, attempts, duplicates) = run(spec, &args, &validated)?;

    let elapsed = start_time.elapsed();
    let rate = emitted as f64 / elapsed.as_secs_f64();
    println!("\nCompleted in {:.2}s", elapsed.as_secs_f64());
    println!(
        "Generated {} mazes at {:.2} mazes/second ({} candidates, {} duplicates skipped)",
        emitted, rate, attempts, duplicates
    );

    Ok(())
}

/// One serialized batch plus the number of mazes it encodes (for accurate progress).
struct Batch {
    bytes: Vec<u8>,
    n: u64,
}

/// The `--fixed-maze` layout: drawn once, from its own reserved PRNG stream.
fn fixed_layout(spec: GenSpec) -> Maze {
    let mut rng = create_instance_prng(spec.seed, spec.generator, spec.solver, FIXED_LAYOUT_ID);
    generators::generate_maze(spec.generator, &mut rng, spec.rows, spec.cols, spec.params)
}

/// Generate and solve a single maze deterministically from its instance id.
fn generate_one(spec: GenSpec, instance_id: u64) -> MazeResult {
    let mut rng = create_instance_prng(spec.seed, spec.generator, spec.solver, instance_id);
    let maze = match spec.fixed {
        Some(layout) => generators::with_random_query(layout, &mut rng),
        None => {
            generators::generate_maze(spec.generator, &mut rng, spec.rows, spec.cols, spec.params)
        }
    };
    let solution = solvers::solve_maze(spec.solver, &maze);
    MazeResult {
        instance_id,
        maze,
        solution,
        generator: spec.generator,
        solver: spec.solver,
        seed: spec.seed,
    }
}

/// Generate a contiguous id range in parallel, preserving ascending id order.
fn gen_chunk(spec: GenSpec, start: u64, end: u64) -> Vec<MazeResult> {
    (start..end)
        .into_par_iter()
        .map(|id| generate_one(spec, id))
        .collect()
}

/// The run's single source of candidate mazes, shared by every partition.
///
/// Candidates are produced in ascending instance-id order and consumed serially, so the output
/// is byte-identical for the same args at any thread count. The seen-set spans the whole run:
/// a layout claimed for eval can never be claimed again for test or train, which is what makes
/// the splits disjoint by construction rather than by hoping two master seeds diverge.
struct CandidateStream<'a> {
    spec: GenSpec<'a>,
    /// Generated-but-unconsumed candidates, ascending id order.
    pending: VecDeque<MazeResult>,
    /// Next instance id to generate.
    next_id: u64,
    /// Hard cap on generated candidates for the whole run.
    max_attempts: u64,
    seen: SeenSet,
    /// Candidates rejected because their layout was already emitted.
    duplicates: u64,
}

impl<'a> CandidateStream<'a> {
    fn new(spec: GenSpec<'a>, max_attempts: u64, expected: u64) -> Self {
        Self {
            spec,
            pending: VecDeque::new(),
            next_id: 0,
            max_attempts,
            seen: SeenSet::with_capacity(expected as usize),
            duplicates: 0,
        }
    }

    /// Generate at least `count` candidates into `pending` without consuming them.
    /// - Used by the balanced path to calibrate the plan-length range from the head of the
    ///   stream; those same candidates are then consumed normally.
    fn prefill(&mut self, count: u64) {
        let have = self.pending.len() as u64;
        if have >= count {
            return;
        }
        let start = self.next_id;
        let end = (start + (count - have)).min(self.max_attempts);
        self.next_id = end;
        self.pending.extend(gen_chunk(self.spec, start, end));
    }

    /// Next candidate in id order, or `None` once the attempt budget is spent.
    fn next_candidate(&mut self) -> Option<MazeResult> {
        loop {
            if let Some(result) = self.pending.pop_front() {
                return Some(result);
            }
            if self.next_id >= self.max_attempts {
                return None;
            }
            let start = self.next_id;
            let end = (start + CHUNK).min(self.max_attempts);
            self.next_id = end;
            self.pending.extend(gen_chunk(self.spec, start, end));
        }
    }

    /// Claim an instance for emission; `false` when it was already emitted or excluded.
    /// - Identity is the layout, or under `--fixed-maze` (one shared layout) the query.
    fn claim(&mut self, maze: &Maze) -> bool {
        let key = match self.spec.fixed {
            Some(_) => query_key(maze),
            None => layout_key(maze),
        };
        if self.seen.claim(key) {
            return true;
        }
        self.duplicates += 1;
        false
    }

    /// Candidates examined so far (generated, minus the unused leftovers).
    fn attempts(&self) -> u64 {
        self.next_id - self.pending.len() as u64
    }
}

/// The calibrated plan-length binning shared by every partition of a balanced run.
struct BinSpec {
    min_len: usize,
    max_len: usize,
    num_bins: usize,
}

impl BinSpec {
    /// Resolve the binning: explicit `--min-len`/`--max-len`, else calibrated from the head of
    /// the candidate stream.
    fn calibrate(stream: &mut CandidateStream, opts: BalancedOpts) -> Result<Self, String> {
        let (min_len, max_len) = match (opts.min_len, opts.max_len) {
            (Some(lo), Some(hi)) => (lo, hi),
            _ => {
                let samples = opts.calibration_samples.max(1);
                println!("Calibrating plan-length range from {} samples...", samples);
                stream.prefill(samples);
                let lens: Vec<usize> = stream
                    .pending
                    .iter()
                    .take(samples as usize)
                    .map(|r| r.solution.path.len())
                    .filter(|&l| l > 0)
                    .collect();
                if lens.is_empty() {
                    return Err("Calibration produced no solvable mazes".into());
                }
                let observed_min = *lens.iter().min().unwrap();
                let observed_max = balanced::percentile(&lens, opts.calibration_trim).unwrap();
                let lo = opts.min_len.unwrap_or(observed_min);
                let hi = opts.max_len.unwrap_or(observed_max.max(lo));
                (lo, hi)
            }
        };
        if max_len < min_len {
            return Err(format!(
                "--max-len ({}) must be >= --min-len ({})",
                max_len, min_len
            ));
        }

        // Clamp num_bins to the integer-length width so every bin is fillable.
        let num_bins = balanced::clamp_num_bins(opts.num_bins, min_len, max_len);
        if num_bins != opts.num_bins {
            eprintln!(
                "Warning: plan-length range [{}, {}] has only {} integer lengths; \
                 clamping --num-bins from {} to {}.",
                min_len,
                max_len,
                max_len - min_len + 1,
                opts.num_bins,
                num_bins
            );
        }
        println!(
            "Balanced sampling: num_bins={} range=[{}, {}]",
            num_bins, min_len, max_len
        );
        Ok(Self {
            min_len,
            max_len,
            num_bins,
        })
    }
}

/// Path of a partition file inside the dataset directory.
fn partition_path(dir: &str, kind: PartitionKind) -> String {
    Path::new(dir)
        .join(format!("{}.jsonl", kind.stem()))
        .to_string_lossy()
        .into_owned()
}

/// Create the dataset directory and refuse to clobber an existing partition file.
///
/// The whole point of a dataset is disjointness; overwriting a split another run depends on
/// would silently break that, so an existing target file is a hard error, not a truncation.
fn prepare_output_dir(
    dir: &str,
    partitions: &[Partition],
) -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all(dir).map_err(|e| format!("--output {dir}: {e}"))?;
    for part in partitions {
        let path = partition_path(dir, part.kind);
        if Path::new(&path).exists() {
            return Err(format!(
                "refusing to overwrite existing partition file {path} \
                 (remove it or choose another --output directory)"
            )
            .into());
        }
    }
    Ok(())
}

/// Dump the parsed CLI args as `args.json` in the dataset directory.
///
/// - Written after the clobber checks, so a refused run leaves the directory untouched.
/// - Overwrites: the file records the invocation that last wrote into the directory.
fn write_args_json(dir: &str, args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(dir).join("args.json");
    let json = serde_json::to_string_pretty(args)?;
    std::fs::write(&path, json + "\n").map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

/// Write the shared layout's wall list as `walls.txt` (`--hide-walls`).
///
/// - Prepending it plus a space to a record's `text` rebuilds the default-placement record.
/// - Never silently replaced: rewriting the same layout is a no-op, a different one is an error
///   (it would no longer describe the partitions already in the directory).
fn write_walls_txt(dir: &str, layout: &Maze) -> Result<(), Box<dyn std::error::Error>> {
    let path = Path::new(dir).join("walls.txt");
    let mut text = serializer::walls_line(layout);
    text.push(b'\n');
    if let Ok(existing) = std::fs::read(&path) {
        if existing == text {
            return Ok(());
        }
        return Err(format!(
            "{} holds a different layout (another --seed/--generator/--rows/--cols?); \
             refusing to overwrite it",
            path.display()
        )
        .into());
    }
    std::fs::write(&path, text).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

/// Pre-seed the seen-set from an existing dataset directory (`--exclude`).
fn load_exclusions(
    stream: &mut CandidateStream,
    dir: &str,
    rows: usize,
    cols: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let paths = exclude::partition_files_in(dir);
    if paths.is_empty() {
        return Err(format!(
            "--exclude {dir}: no eval.jsonl / test.jsonl / train.jsonl found \
             (excluding nothing is never what you meant)"
        )
        .into());
    }
    let started = Instant::now();
    let mut excluded: u64 = 0;
    let stats = exclude::scan_files(&paths, rows, cols, |key| {
        if stream.seen.claim(key) {
            excluded += 1;
        }
    })?;
    let skipped = match stats.skipped_dims {
        0 => String::new(),
        n => format!(", {n} not {rows}x{cols}"),
    };
    eprintln!(
        "Excluding {excluded} layouts read from {} file(s) ({} lines{skipped}) in {:.2}s",
        paths.len(),
        stats.lines,
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Spawn the writer thread; payload carries a maze count for accurate progress.
fn spawn_writer(
    output_path: String,
    total: u64,
) -> (
    SyncSender<Batch>,
    thread::JoinHandle<Result<(), std::io::Error>>,
) {
    let (tx, rx) = sync_channel::<Batch>(100);
    let progress = ProgressBar::new(total);
    progress.set_style(
        ProgressStyle::default_bar()
            .template(
                "[{elapsed_precise}] {bar:40.green/red} {pos}/{len} mazes written ({per_sec})",
            )
            .expect("valid progress template")
            .progress_chars("##-"),
    );
    let handle = thread::spawn(move || -> Result<(), std::io::Error> {
        let file = File::create(&output_path)?;
        let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);
        let mut total_written = 0u64;
        for batch in rx {
            writer.write_all(&batch.bytes)?;
            total_written += batch.n;
            progress.set_position(total_written);
        }
        writer.flush()?;
        progress.finish_with_message("All mazes written!");
        Ok(())
    });
    (tx, handle)
}

/// Serialize and send a batch, clearing the buffer.
fn flush_batch(tx: &SyncSender<Batch>, buffer: &mut Vec<MazeResult>, placement: QueryPlacement) {
    if buffer.is_empty() {
        return;
    }
    let bytes = process_batch(buffer, placement);
    tx.send(Batch {
        bytes,
        n: buffer.len() as u64,
    })
    .expect("writer thread alive");
    buffer.clear();
}

/// What one partition managed to emit, and how its bins filled up.
struct PartitionOutcome {
    emitted: u64,
    caps: Vec<u64>,
    filled: Vec<u64>,
}

/// Generate one partition file from the shared candidate stream.
///
/// Acceptance order: bin first, then layout. A candidate rejected for a full bin keeps its
/// layout available for a later partition; only an emitted maze burns its layout.
fn generate_partition(
    stream: &mut CandidateStream,
    part: Partition,
    path: &str,
    bins: Option<&BinSpec>,
    placement: QueryPlacement,
) -> Result<PartitionOutcome, Box<dyn std::error::Error>> {
    let caps = match bins {
        Some(b) => balanced::compute_caps(part.count, b.num_bins),
        None => vec![part.count],
    };
    let mut filled = vec![0u64; caps.len()];
    let mut emitted: u64 = 0;

    let (tx, writer_handle) = spawn_writer(path.to_string(), part.count);
    let mut out_buffer: Vec<MazeResult> = Vec::with_capacity(BATCH_SIZE);

    while emitted < part.count {
        let Some(result) = stream.next_candidate() else {
            break;
        };
        // Only a `--fixed-maze` searchformer layout (not validated per query) can be unsolvable.
        if result.solution.path.is_empty() {
            continue;
        }
        let bin = match bins {
            Some(b) => {
                match balanced::bin_index(
                    result.solution.path.len(),
                    b.min_len,
                    b.max_len,
                    b.num_bins,
                ) {
                    Some(bin) => bin,
                    None => continue,
                }
            }
            None => 0,
        };
        if filled[bin] >= caps[bin] {
            continue;
        }
        if !stream.claim(&result.maze) {
            continue;
        }
        filled[bin] += 1;
        emitted += 1;
        out_buffer.push(result);
        if out_buffer.len() >= BATCH_SIZE {
            flush_batch(&tx, &mut out_buffer, placement);
        }
    }

    flush_batch(&tx, &mut out_buffer, placement);
    drop(tx);
    writer_handle.join().expect("writer thread joins")?;

    Ok(PartitionOutcome {
        emitted,
        caps,
        filled,
    })
}

/// Report per-bin deficits for a partition that could not be filled; no silent truncation.
fn report_underfill(
    part: Partition,
    outcome: &PartitionOutcome,
    bins: Option<&BinSpec>,
    attempts: u64,
    duplicates: u64,
) {
    eprintln!(
        "\nUnder-filled {} after {} candidates (emitted {}/{}, {} duplicates skipped):",
        part.kind.stem(),
        attempts,
        outcome.emitted,
        part.count,
        duplicates
    );
    let Some(bins) = bins else {
        return;
    };
    let ranges = balanced::bin_ranges(bins.min_len, bins.max_len, bins.num_bins);
    eprintln!("  bin  range        cap   filled  deficit");
    for (b, bin_range) in ranges.iter().enumerate() {
        let range = match bin_range {
            Some((lo, hi)) => format!("[{}, {}]", lo, hi),
            None => "-".to_string(),
        };
        let deficit = outcome.caps[b].saturating_sub(outcome.filled[b]);
        eprintln!(
            "  {:>3}  {:<11}  {:>4}   {:>6}  {:>7}",
            b, range, outcome.caps[b], outcome.filled[b], deficit
        );
    }
}

/// Generate the whole dataset: every requested partition in the fixed eval → test → train
/// order, sharing one candidate stream and seen-set so the splits are mutually layout-disjoint.
///
/// Returns `(emitted, candidates generated, duplicate layouts skipped)`.
fn run(
    spec: GenSpec,
    args: &Args,
    validated: &Validated,
) -> Result<(u64, u64, u64), Box<dyn std::error::Error>> {
    prepare_output_dir(&validated.output_dir, &validated.partitions)?;
    // `--hide-walls` requires `--fixed-maze`, so the layout is always there.
    if let (QueryPlacement::QueryOnly, Some(layout)) = (validated.query_placement, spec.fixed) {
        write_walls_txt(&validated.output_dir, layout)?;
    }
    write_args_json(&validated.output_dir, args)?;

    let mut stream = CandidateStream::new(spec, validated.max_attempts, validated.total_count());
    if let Some(dir) = &validated.exclude_dir {
        load_exclusions(&mut stream, dir, validated.rows, validated.cols)?;
    }

    let bins = match validated.balanced {
        Some(opts) => Some(BinSpec::calibrate(&mut stream, opts)?),
        None => None,
    };

    let mut total: u64 = 0;
    for &part in &validated.partitions {
        let path = partition_path(&validated.output_dir, part.kind);
        println!("  {} -> {path} ({} mazes)...", part.kind.stem(), part.count);
        let outcome = generate_partition(
            &mut stream,
            part,
            &path,
            bins.as_ref(),
            validated.query_placement,
        )?;
        total += outcome.emitted;
        if outcome.emitted < part.count {
            report_underfill(
                part,
                &outcome,
                bins.as_ref(),
                stream.attempts(),
                stream.duplicates,
            );
            if !validated.allow_underfill {
                // Tailor the hint: a balanced run usually starves a bin, an unbalanced one has
                // run out of distinct layouts for this grid.
                let hint = match bins {
                    Some(_) => {
                        "Reduce --num-bins, lower --max-len, or raise --max-attempts \
                         (or pass --allow-underfill)."
                    }
                    None => {
                        "The grid admits too few distinct layouts; enlarge --rows/--cols, \
                         lower the counts, or raise --max-attempts (or pass --allow-underfill)."
                    }
                };
                return Err(format!(
                    "Under-filled {}: {}/{} mazes after {} candidates. {hint}",
                    part.kind.stem(),
                    outcome.emitted,
                    part.count,
                    stream.attempts()
                )
                .into());
            }
            // The stream is spent; later partitions cannot do better.
            break;
        }
    }

    Ok((total, stream.attempts(), stream.duplicates))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walls_txt_is_written_once_and_never_replaced_by_another_layout() {
        let dir = std::env::temp_dir().join(format!("maze_walls_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dir_str = dir.to_str().unwrap();
        let mut a = Maze::all_floor(4, 4);
        a.set_cell(1, 0, false);
        let mut b = a.clone();
        b.set_cell(2, 3, false);

        write_walls_txt(dir_str, &a).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("walls.txt")).unwrap(),
            "wall 1 0\n"
        );
        // Same layout again (e.g. a later run adding another split): fine.
        write_walls_txt(dir_str, &a).unwrap();
        // A different layout would orphan the records already written.
        assert!(write_walls_txt(dir_str, &b).is_err());
        assert_eq!(
            std::fs::read_to_string(dir.join("walls.txt")).unwrap(),
            "wall 1 0\n"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn fixed_maze_shares_the_layout_and_never_repeats_a_query() {
        let params = GeneratorParams::from_vec(Vec::new()).unwrap();
        let mut spec = GenSpec {
            seed: 0,
            generator: GeneratorType::Wilson,
            solver: SolverType::AStar,
            rows: 5,
            cols: 5,
            params: &params,
            fixed: None,
        };
        let layout = fixed_layout(spec);
        spec.fixed = Some(&layout);
        let n = generators::floors(&layout).len();

        // One stream spans every split, so "no repeat in the stream" = none across splits.
        let mut stream = CandidateStream::new(spec, 20_000, 0);
        let mut emitted = 0;
        while let Some(r) = stream.next_candidate() {
            assert_eq!(layout_key(&r.maze), layout_key(&layout));
            assert_ne!(r.maze.start, r.maze.goal);
            assert!(!r.solution.path.is_empty());
            emitted += stream.claim(&r.maze) as usize;
        }
        // Every ordered pair of distinct floor cells is drawn, and claimed exactly once.
        assert_eq!(emitted, n * (n - 1));
    }
}
