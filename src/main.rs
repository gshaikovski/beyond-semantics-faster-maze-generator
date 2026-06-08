mod balanced;
mod generators;
mod parameters;
mod prng;
mod serializer;
mod solvers;
mod types;

use clap::{Parser, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::mpsc::{sync_channel, SyncSender};
use std::thread;
use std::time::Instant;

use crate::parameters::{print_all_params_help, print_param_help, GeneratorParams};
use crate::prng::create_instance_prng;
use crate::serializer::process_batch;
use crate::types::MazeResult;

#[derive(Clone, Copy, Debug, ValueEnum, Hash)]
pub enum GeneratorType {
    Dfs,
    Kruskal,
    Wilson,
    Searchformer,
    DrunkardsWalk,
}

#[derive(Clone, Copy, Debug, ValueEnum, Hash)]
pub enum SolverType {
    #[value(name = "astar")]
    AStar,
}

#[derive(Parser)]
#[command(name = "maze_gen_fast")]
#[command(about = "Parallel maze generation with hierarchical PRNG")]
struct Args {
    /// Generator algorithm
    #[arg(short, long, value_enum, required_unless_present = "list_params")]
    generator: Option<GeneratorType>,

    /// Solver algorithm
    #[arg(short, long, value_enum, required_unless_present = "list_params")]
    solver: Option<SolverType>,

    /// Master seed for PRNG
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Number of mazes to generate
    #[arg(short, long, required_unless_present = "list_params")]
    count: Option<u64>,

    /// Maze height
    #[arg(long, default_value = "30")]
    rows: usize,

    /// Maze width
    #[arg(long, default_value = "30")]
    cols: usize,

    /// Output file
    #[arg(short, long, default_value = "output.jsonl")]
    output: String,

    /// Number of threads (defaults to all cores)
    #[arg(short, long)]
    threads: Option<usize>,

    /// Generator parameters as key=value pairs
    #[arg(long = "param", value_parser = parse_key_val::<String, String>)]
    params: Vec<(String, String)>,

    /// List parameters for a specific generator or all generators
    #[arg(long, value_name = "GENERATOR")]
    list_params: Option<Option<GeneratorType>>,

    /// Sample mazes with a uniform plan-length distribution (rejection sampling)
    #[arg(long)]
    balanced: bool,

    /// Number of equal-width plan-length bins (only with --balanced)
    #[arg(long, default_value = "10")]
    num_bins: usize,

    /// Minimum plan length for binning (only with --balanced; default: calibrated)
    #[arg(long)]
    min_len: Option<usize>,

    /// Maximum plan length for binning (only with --balanced; default: calibrated)
    #[arg(long)]
    max_len: Option<usize>,

    /// Calibration sample size used to auto-detect the length range (only with --balanced)
    #[arg(long, default_value = "10000")]
    calibration_samples: u64,

    /// Upper-percentile trim for the calibrated max length, in [0, 1] (only with --balanced)
    #[arg(long, default_value = "0.99")]
    calibration_trim: f64,

    /// Max generation attempts before giving up on unfilled bins (only with --balanced)
    #[arg(long)]
    max_attempts: Option<u64>,

    /// Exit 0 even if some bins could not be filled (only with --balanced)
    #[arg(long)]
    allow_underfill: bool,
}

/// Parse key=value pairs
fn parse_key_val<T, U>(
    s: &str,
) -> Result<(T, U), Box<dyn std::error::Error + Send + Sync + 'static>>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
    U: std::str::FromStr,
    U::Err: std::error::Error + Send + Sync + 'static,
{
    let pos = s
        .find('=')
        .ok_or_else(|| format!("invalid KEY=value: no `=` found in `{}`", s))?;
    Ok((s[..pos].parse()?, s[pos + 1..].parse()?))
}

/// Batch size for serialization / sending to the writer.
const BATCH_SIZE: usize = 1000;

/// Shared maze-generation parameters; threaded through both run paths.
#[derive(Clone, Copy)]
struct GenSpec<'a> {
    seed: u64,
    generator: GeneratorType,
    solver: SolverType,
    rows: usize,
    cols: usize,
    params: &'a GeneratorParams,
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

    // Extract required args (safe because of required_unless_present)
    let generator = args.generator.expect("generator required");
    let solver = args.solver.expect("solver required");
    let count = args.count.expect("count required");

    // Parse generator parameters
    let generator_params = GeneratorParams::from_vec(args.params)?;

    // Set thread pool size if specified
    if let Some(threads) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()?;
    }

    // Start timing
    let start_time = Instant::now();
    println!("Generating {} mazes...", count);

    let spec = GenSpec {
        seed: args.seed,
        generator,
        solver,
        rows: args.rows,
        cols: args.cols,
        params: &generator_params,
    };

    if args.balanced {
        let opts = BalancedOpts {
            num_bins: args.num_bins,
            min_len: args.min_len,
            max_len: args.max_len,
            calibration_samples: args.calibration_samples,
            calibration_trim: args.calibration_trim,
            max_attempts: args.max_attempts,
            allow_underfill: args.allow_underfill,
        };
        run_balanced(spec, count, args.output, opts)?;
    } else {
        run_standard(spec, count, args.output)?;
    }

    let elapsed = start_time.elapsed();
    let rate = count as f64 / elapsed.as_secs_f64();

    println!("\nCompleted in {:.2}s", elapsed.as_secs_f64());
    println!("Generated {} mazes at {:.2} mazes/second", count, rate);

    Ok(())
}

/// One serialized batch plus the number of mazes it encodes (for accurate progress).
struct Batch {
    bytes: Vec<u8>,
    n: u64,
}

/// Generate and solve a single maze deterministically from its instance id.
fn generate_one(spec: GenSpec, instance_id: u64) -> MazeResult {
    let mut rng = create_instance_prng(spec.seed, spec.generator, spec.solver, instance_id);
    let maze =
        generators::generate_maze(spec.generator, &mut rng, spec.rows, spec.cols, spec.params);
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
            writer.flush()?;
        }
        writer.flush()?;
        progress.finish_with_message("All mazes written!");
        Ok(())
    });
    (tx, handle)
}

/// Standard path: generate exactly `count` mazes in id order.
/// - Output is identical to the pre-`--balanced` behaviour.
fn run_standard(
    spec: GenSpec,
    count: u64,
    output: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let (tx, writer_handle) = spawn_writer(output, count);

    // Process mazes in batches
    (0..count)
        .step_by(BATCH_SIZE)
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each_with(tx, |tx, batch_start| {
            let batch_end = (batch_start + BATCH_SIZE as u64).min(count);
            let mut batch_results: Vec<MazeResult> = Vec::with_capacity(BATCH_SIZE);
            for instance_id in batch_start..batch_end {
                batch_results.push(generate_one(spec, instance_id));
            }
            let bytes = process_batch(&batch_results);
            tx.send(Batch {
                bytes,
                n: batch_results.len() as u64,
            })
            .unwrap();
        });

    writer_handle.join().unwrap()?;
    Ok(())
}

/// Options for the balanced sampling path.
struct BalancedOpts {
    num_bins: usize,
    min_len: Option<usize>,
    max_len: Option<usize>,
    calibration_samples: u64,
    calibration_trim: f64,
    max_attempts: Option<u64>,
    allow_underfill: bool,
}

/// Balanced path: rejection-sample to a uniform plan-length histogram.
/// - Deterministic: candidates accepted in ascending id order, independent of thread count.
/// - Total output == count when feasible; otherwise reports per-bin deficits.
fn run_balanced(
    spec: GenSpec,
    count: u64,
    output: String,
    opts: BalancedOpts,
) -> Result<(), Box<dyn std::error::Error>> {
    // Generation chunk size: bounds wasted work past target while keeping parallelism high.
    const CHUNK: u64 = 8 * BATCH_SIZE as u64;

    if opts.num_bins < 1 {
        return Err("--num-bins must be >= 1".into());
    }

    // Order-preserving parallel generation of a contiguous id range.
    let gen_chunk = |start: u64, end: u64| -> Vec<MazeResult> {
        (start..end)
            .into_par_iter()
            .map(|id| generate_one(spec, id))
            .collect()
    };

    // Calibrate the plan-length range; reuse the samples as the first candidates.
    let mut pending: Vec<MazeResult> = Vec::new();
    let mut next_id: u64 = 0;
    let (min_len, max_len) = match (opts.min_len, opts.max_len) {
        (Some(lo), Some(hi)) => (lo, hi),
        _ => {
            let s = opts.calibration_samples.max(1);
            println!("Calibrating plan-length range from {} samples...", s);
            pending = gen_chunk(0, s);
            next_id = s;
            let lens: Vec<usize> = pending
                .iter()
                .map(|r| r.solution.path.len())
                .filter(|&l| l > 0)
                .collect();
            if lens.is_empty() {
                return Err("Calibration produced no solvable mazes".into());
            }
            let obs_min = *lens.iter().min().unwrap();
            let obs_max = balanced::percentile(&lens, opts.calibration_trim).unwrap();
            let lo = opts.min_len.unwrap_or(obs_min);
            let hi = opts.max_len.unwrap_or(obs_max.max(lo));
            (lo, hi)
        }
    };

    if max_len < min_len {
        return Err(format!("--max-len ({}) must be >= --min-len ({})", max_len, min_len).into());
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

    let caps = balanced::compute_caps(count, num_bins);
    let max_attempts = opts.max_attempts.unwrap_or_else(|| {
        count
            .saturating_mul(num_bins as u64)
            .saturating_mul(100)
            .max(count)
    });

    println!(
        "Balanced sampling: count={} num_bins={} range=[{}, {}] max_attempts={}",
        count, num_bins, min_len, max_len, max_attempts
    );

    let (tx, writer_handle) = spawn_writer(output, count);

    // Serial, deterministic accept state.
    let mut filled = vec![0u64; num_bins];
    let mut total_accepted: u64 = 0;
    let mut out_buffer: Vec<MazeResult> = Vec::with_capacity(BATCH_SIZE);

    'outer: loop {
        // Refill candidates in ascending id order if needed.
        if pending.is_empty() {
            if next_id >= max_attempts {
                break;
            }
            let start = next_id;
            let end = (start + CHUNK).min(max_attempts);
            next_id = end;
            pending = gen_chunk(start, end);
        }

        // Serial accept pass (pending is in ascending id order).
        for result in pending.drain(..) {
            let len = result.solution.path.len();
            if let Some(b) = balanced::bin_index(len, min_len, max_len, num_bins) {
                if filled[b] < caps[b] {
                    filled[b] += 1;
                    total_accepted += 1;
                    out_buffer.push(result);
                    if out_buffer.len() >= BATCH_SIZE {
                        let bytes = process_batch(&out_buffer);
                        tx.send(Batch {
                            bytes,
                            n: out_buffer.len() as u64,
                        })
                        .unwrap();
                        out_buffer.clear();
                    }
                    if total_accepted >= count {
                        break 'outer;
                    }
                }
            }
        }
    }

    // Flush remaining accepted mazes.
    if !out_buffer.is_empty() {
        let bytes = process_batch(&out_buffer);
        tx.send(Batch {
            bytes,
            n: out_buffer.len() as u64,
        })
        .unwrap();
    }
    drop(tx);
    writer_handle.join().unwrap()?;

    // Under-fill handling: report per-bin deficits; no silent truncation.
    if total_accepted < count {
        let ranges = balanced::bin_ranges(min_len, max_len, num_bins);
        eprintln!(
            "\nUnder-filled after {} attempts (accepted {}/{}):",
            next_id, total_accepted, count
        );
        eprintln!("  bin  range        cap   filled  deficit");
        for b in 0..num_bins {
            let range = match ranges[b] {
                Some((lo, hi)) => format!("[{}, {}]", lo, hi),
                None => "-".to_string(),
            };
            let deficit = caps[b].saturating_sub(filled[b]);
            eprintln!(
                "  {:>3}  {:<11}  {:>4}   {:>6}  {:>7}",
                b, range, caps[b], filled[b], deficit
            );
        }
        if !opts.allow_underfill {
            return Err(format!(
                "Balanced sampling under-filled: {}/{} mazes after {} attempts. \
                 Reduce --num-bins, lower --max-len, or raise --max-attempts \
                 (or pass --allow-underfill).",
                total_accepted, count, next_id
            )
            .into());
        }
    }

    Ok(())
}
