//! Command-line interface: argument parsing, validation, and dataset partitions.

use crate::serializer::QueryPlacement;
use crate::{GeneratorType, SolverType};
use clap::Parser;
use serde::Serialize;
use std::path::{Path, PathBuf};

/// A dataset partition. Order of the variants is the hardcoded generation order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionKind {
    Eval,
    Test,
    Train,
}

impl PartitionKind {
    /// Generation order: eval, then test, then train. One candidate stream and one seen-set
    /// span the whole run, so each later split is layout-disjoint from the earlier ones.
    pub const ORDER: [PartitionKind; 3] = [
        PartitionKind::Eval,
        PartitionKind::Test,
        PartitionKind::Train,
    ];

    /// Output file stem (`<stem>.jsonl`) inside the dataset directory.
    pub fn stem(self) -> &'static str {
        match self {
            PartitionKind::Eval => "eval",
            PartitionKind::Test => "test",
            PartitionKind::Train => "train",
        }
    }
}

/// One requested partition and its exact maze count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Partition {
    pub kind: PartitionKind,
    pub count: u64,
}

/// Balanced (uniform plan-length) sampling options.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BalancedOpts {
    pub num_bins: usize,
    pub min_len: Option<usize>,
    pub max_len: Option<usize>,
    pub calibration_samples: u64,
    pub calibration_trim: f64,
}

/// Validated configuration derived from `Args`.
#[derive(Debug)]
pub struct Validated {
    pub generator: GeneratorType,
    pub solver: SolverType,
    pub rows: usize,
    pub cols: usize,
    /// Requested partitions in generation order (eval, test, train), absent ones dropped.
    pub partitions: Vec<Partition>,
    /// Dataset directory the partition files and `args.json` are written into.
    pub output_dir: String,
    /// Uniform plan-length sampling (None = accept every candidate).
    pub balanced: Option<BalancedOpts>,
    /// Hard cap on candidate mazes generated across the whole run.
    pub max_attempts: u64,
    /// Exit 0 even when a partition could not be filled.
    pub allow_underfill: bool,
    /// Dataset directory whose layouts must not be re-emitted (None when `--exclude` absent).
    pub exclude_dir: Option<String>,
    /// Where the `query start ... goal ...` block sits in the emitted `text`.
    pub query_placement: QueryPlacement,
    /// One wall layout for every instance of every split; only the query varies.
    pub fixed_maze: bool,
}

impl Validated {
    /// Total mazes requested across all partitions.
    pub fn total_count(&self) -> u64 {
        self.partitions.iter().map(|p| p.count).sum()
    }
}

/// Validated configuration for `--puzzle-file`.
#[derive(Debug, PartialEq, Eq)]
pub struct ResolveOpts {
    /// Puzzle files to re-solve, in the order given.
    pub inputs: Vec<PathBuf>,
    /// Solver to re-solve with; the record's own `solver` field is not consulted.
    pub solver: SolverType,
}

/// What this invocation does. Exactly one of the two, decided by `--puzzle-file`.
#[derive(Debug)]
pub enum Mode {
    /// Generate a dataset directory.
    Generate(Box<Validated>),
    /// Re-solve an existing puzzle file.
    Resolve(ResolveOpts),
}

#[derive(Parser, Serialize)]
#[command(name = "maze_gen_fast")]
#[command(about = "Parallel maze generation with hierarchical PRNG")]
pub struct Args {
    /// Generator algorithm
    #[arg(
        short,
        long,
        value_enum,
        required_unless_present_any = ["list_params", "puzzle_file"]
    )]
    pub generator: Option<GeneratorType>,

    /// Solver algorithm
    #[arg(
        short,
        long,
        value_enum,
        required_unless_present_any = ["list_params", "puzzle_file"]
    )]
    pub solver: Option<SolverType>,

    /// Master seed for PRNG
    #[arg(long, default_value = "42")]
    pub seed: u64,

    /// Maze height
    #[arg(long, default_value = "30")]
    pub rows: usize,

    /// Maze width
    #[arg(long, default_value = "30")]
    pub cols: usize,

    /// Dataset directory; partition files (eval.jsonl, test.jsonl, train.jsonl) are written here
    #[arg(short, long, required_unless_present_any = ["list_params", "puzzle_file"])]
    pub output: Option<String>,

    /// Number of eval mazes; omit to skip the split
    #[arg(long)]
    pub eval: Option<u64>,

    /// Number of test mazes; omit to skip the split
    #[arg(long)]
    pub test: Option<u64>,

    /// Number of train mazes; omit to skip the split
    #[arg(long)]
    pub train: Option<u64>,

    /// Number of threads (defaults to all cores)
    #[arg(short, long)]
    pub threads: Option<usize>,

    /// Generator parameters as key=value pairs
    #[arg(long = "param", value_parser = parse_key_val::<String, String>)]
    pub params: Vec<(String, String)>,

    /// List parameters for a specific generator or all generators
    #[arg(long, value_name = "GENERATOR")]
    #[serde(skip)]
    pub list_params: Option<Option<GeneratorType>>,

    /// Sample mazes with a uniform plan-length distribution (rejection sampling)
    #[arg(long)]
    pub balanced: bool,

    /// Number of equal-width plan-length bins (only with --balanced)
    #[arg(long, default_value = "10")]
    pub num_bins: usize,

    /// Minimum plan length for binning (only with --balanced; default: calibrated)
    #[arg(long)]
    pub min_len: Option<usize>,

    /// Maximum plan length for binning (only with --balanced; default: calibrated)
    #[arg(long)]
    pub max_len: Option<usize>,

    /// Calibration sample size used to auto-detect the length range (only with --balanced)
    #[arg(long, default_value = "10000")]
    pub calibration_samples: u64,

    /// Upper-percentile trim for the calibrated max length, in [0, 1] (only with --balanced)
    #[arg(long, default_value = "0.99")]
    pub calibration_trim: f64,

    /// Max candidate mazes to generate before giving up on unfilled partitions
    #[arg(long)]
    pub max_attempts: Option<u64>,

    /// Exit 0 even if some partition could not be filled
    #[arg(long)]
    pub allow_underfill: bool,

    /// Existing dataset directory to stay disjoint from: no wall layout appearing in any of its
    /// partition files is emitted again (matched by walls alone, whatever the start/goal)
    #[arg(long)]
    pub exclude: Option<String>,

    /// Emit the legacy text layout, `query start ... goal ...` before the wall list
    /// (default: the query block follows the walls, just before `reasoning`)
    #[arg(long)]
    pub legacy_query_first: bool,

    /// Omit the wall list from `text`: the prompt is just `query start ... goal ...`
    #[arg(long, conflicts_with = "legacy_query_first")]
    pub hide_walls: bool,

    /// Use one wall layout for every instance in every split; each instance draws its own
    /// start/goal, and no (start, goal) pair repeats within or across splits
    #[arg(long, conflicts_with = "exclude")]
    pub fixed_maze: bool,

    /// Re-solve an existing maze JSONL instead of generating one: writes
    /// <dir>/<stem>_resolved.jsonl with a fresh A* trace and optimal plan per record, every
    /// other JSON key copied verbatim
    #[arg(long, conflicts_with_all = [
        "list_params", "generator", "output", "eval", "test", "train",
        "balanced", "exclude", "params", "legacy_query_first",
        "max_attempts", "allow_underfill", "fixed_maze", "hide_walls",
    ])]
    pub puzzle_file: Option<String>,
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

/// Smallest grid side every generator can lay out with both parity offsets (see `validate`).
const MIN_GRID: usize = 4;

/// Attempts allowed per requested maze before a run is declared under-filled.
/// - Balanced sampling rejects most candidates, so its budget scales with the bin count.
const ATTEMPTS_PER_MAZE: u64 = 100;

impl Args {
    /// What this invocation does: re-solve a puzzle file, or generate a dataset.
    pub fn mode(&self) -> Result<Mode, String> {
        match self.puzzle_file.as_deref() {
            Some(path) => Ok(Mode::Resolve(self.validate_resolve(path)?)),
            None => Ok(Mode::Generate(Box::new(self.validate()?))),
        }
    }

    /// Validate a `--puzzle-file` run.
    ///
    /// Deliberately shares nothing with `validate`: there are no partitions, no `--output`, and
    /// no `MIN_GRID` floor (that exists because the lattice generators spin forever on a narrow
    /// grid — a small *record* re-solves fine). Dimensions come from each record.
    fn validate_resolve(&self, path: &str) -> Result<ResolveOpts, String> {
        if path.trim().is_empty() {
            return Err("--puzzle-file must be a JSONL file path".into());
        }
        let input = Path::new(path);
        if input.is_dir() {
            return Err(format!(
                "--puzzle-file {path} is a directory; pass one JSONL file"
            ));
        }
        if !input.is_file() {
            return Err(format!("--puzzle-file {path}: no such file"));
        }
        Ok(ResolveOpts {
            inputs: vec![input.to_path_buf()],
            // Only one solver exists; the record's own `solver` field is copied, not read.
            solver: self.solver.unwrap_or(SolverType::AStar),
        })
    }

    /// Validate inputs; resolve the requested partitions and the sampling mode.
    pub fn validate(&self) -> Result<Validated, String> {
        let generator = self.generator.ok_or("--generator is required")?;
        let solver = self.solver.ok_or("--solver is required")?;
        // wilson/dfs/kruskal lay rooms on a 2-step parity lattice whose offset is drawn per
        // instance. Below 4 the odd-offset lattice holds a single room, so the maze has one
        // floor cell and no distinct start/goal — those generators then spin forever looking
        // for one. Refuse the shape rather than hang.
        if self.rows < MIN_GRID || self.cols < MIN_GRID {
            return Err(format!("--rows and --cols must be >= {MIN_GRID}"));
        }
        let output_dir = match self.output.as_deref() {
            None | Some("") => return Err("--output must be a dataset directory path".into()),
            Some(s) if s.trim().is_empty() => {
                return Err("--output must be a dataset directory path".into())
            }
            Some(s) => s.to_string(),
        };
        let partitions = self.partitions()?;
        let balanced = self.balanced_opts()?;
        let exclude_dir = match self.exclude.as_deref() {
            Some(s) if s.trim().is_empty() => {
                return Err("--exclude must be a non-empty dataset directory path".into())
            }
            Some(s) => Some(s.to_string()),
            None => None,
        };

        let total: u64 = partitions.iter().map(|p| p.count).sum();
        let bin_factor = balanced.map_or(1, |b| b.num_bins) as u64;
        let max_attempts = match self.max_attempts {
            Some(0) => return Err("--max-attempts must be >= 1".into()),
            Some(n) => n,
            None => total
                .saturating_mul(bin_factor)
                .saturating_mul(ATTEMPTS_PER_MAZE)
                .max(total),
        };

        Ok(Validated {
            generator,
            solver,
            rows: self.rows,
            cols: self.cols,
            partitions,
            output_dir,
            balanced,
            max_attempts,
            allow_underfill: self.allow_underfill,
            exclude_dir,
            query_placement: match (self.hide_walls, self.legacy_query_first) {
                (true, _) => QueryPlacement::QueryOnly,
                (false, true) => QueryPlacement::First,
                (false, false) => QueryPlacement::AfterWalls,
            },
            fixed_maze: self.fixed_maze,
        })
    }

    /// Requested partitions in generation order; at least one is required and each count >= 1.
    fn partitions(&self) -> Result<Vec<Partition>, String> {
        let mut out = Vec::new();
        for kind in PartitionKind::ORDER {
            let count = match kind {
                PartitionKind::Eval => self.eval,
                PartitionKind::Test => self.test,
                PartitionKind::Train => self.train,
            };
            match count {
                None => continue,
                Some(0) => return Err(format!("--{} must be >= 1", kind.stem())),
                Some(count) => out.push(Partition { kind, count }),
            }
        }
        if out.is_empty() {
            return Err("at least one of --eval, --test, --train is required".into());
        }
        Ok(out)
    }

    /// Balanced options when `--balanced` is set, `None` otherwise.
    fn balanced_opts(&self) -> Result<Option<BalancedOpts>, String> {
        if !self.balanced {
            return Ok(None);
        }
        if self.num_bins < 1 {
            return Err("--num-bins must be >= 1".into());
        }
        if !(0.0..=1.0).contains(&self.calibration_trim) {
            return Err("--calibration-trim must be in [0, 1]".into());
        }
        if let (Some(lo), Some(hi)) = (self.min_len, self.max_len) {
            if hi < lo {
                return Err(format!("--max-len ({hi}) must be >= --min-len ({lo})"));
            }
        }
        Ok(Some(BalancedOpts {
            num_bins: self.num_bins,
            min_len: self.min_len,
            max_len: self.max_len,
            calibration_samples: self.calibration_samples,
            calibration_trim: self.calibration_trim,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an `Args` with only the fields a test cares about; the rest take valid defaults.
    fn args(eval: Option<u64>, test: Option<u64>, train: Option<u64>) -> Args {
        Args {
            generator: Some(GeneratorType::Wilson),
            solver: Some(SolverType::AStar),
            seed: 0,
            rows: 10,
            cols: 10,
            output: Some("ds".into()),
            eval,
            test,
            train,
            threads: None,
            params: Vec::new(),
            list_params: None,
            balanced: false,
            num_bins: 10,
            min_len: None,
            max_len: None,
            calibration_samples: 10000,
            calibration_trim: 0.99,
            max_attempts: None,
            allow_underfill: false,
            exclude: None,
            legacy_query_first: false,
            hide_walls: false,
            fixed_maze: false,
            puzzle_file: None,
        }
    }

    #[test]
    fn partitions_built_in_fixed_order() {
        // CLI order is irrelevant: always eval, test, train.
        let v = args(Some(10), Some(20), Some(30)).validate().unwrap();
        assert_eq!(
            v.partitions,
            vec![
                Partition {
                    kind: PartitionKind::Eval,
                    count: 10
                },
                Partition {
                    kind: PartitionKind::Test,
                    count: 20
                },
                Partition {
                    kind: PartitionKind::Train,
                    count: 30
                },
            ]
        );
        assert_eq!(v.total_count(), 60);
    }

    #[test]
    fn absent_partitions_are_skipped() {
        let v = args(None, None, Some(5)).validate().unwrap();
        assert_eq!(v.partitions.len(), 1);
        assert_eq!(v.partitions[0].kind, PartitionKind::Train);
    }

    #[test]
    fn requires_at_least_one_partition() {
        assert!(args(None, None, None).validate().is_err());
    }

    #[test]
    fn zero_count_rejected() {
        assert!(args(Some(0), None, None).validate().is_err());
    }

    #[test]
    fn degenerate_grid_rejected() {
        let mut a = args(None, None, Some(5));
        a.rows = MIN_GRID - 1;
        assert!(a.validate().is_err());
        a.rows = MIN_GRID;
        assert!(a.validate().is_ok());
    }

    #[test]
    fn output_directory_required() {
        let mut a = args(None, None, Some(5));
        a.output = Some("  ".into());
        assert!(a.validate().is_err());
    }

    #[test]
    fn balanced_opts_only_when_flag_set() {
        // The balanced knobs are inert without --balanced.
        let mut a = args(None, None, Some(5));
        a.num_bins = 0;
        assert_eq!(a.validate().unwrap().balanced, None);
        a.balanced = true;
        assert!(a.validate().is_err());
    }

    #[test]
    fn balanced_range_and_trim_validated() {
        let mut a = args(None, None, Some(5));
        a.balanced = true;
        a.min_len = Some(20);
        a.max_len = Some(10);
        assert!(a.validate().unwrap_err().contains("--max-len"));
        a.max_len = Some(30);
        assert!(a.validate().is_ok());
        a.calibration_trim = 1.5;
        assert!(a.validate().unwrap_err().contains("--calibration-trim"));
    }

    #[test]
    fn query_follows_the_walls_unless_the_legacy_flag_is_set() {
        let mut a = args(None, None, Some(5));
        assert_eq!(
            a.validate().unwrap().query_placement,
            QueryPlacement::AfterWalls
        );
        a.legacy_query_first = true;
        assert_eq!(a.validate().unwrap().query_placement, QueryPlacement::First);
        a.legacy_query_first = false;
        a.hide_walls = true;
        assert_eq!(
            a.validate().unwrap().query_placement,
            QueryPlacement::QueryOnly
        );
    }

    #[test]
    fn puzzle_file_needs_no_generate_arguments() {
        // The whole point of the mode: `--puzzle-file <path>` on its own is a complete command.
        let path = std::env::temp_dir().join(format!("maze_cli_{}.jsonl", std::process::id()));
        std::fs::write(&path, "").unwrap();
        let argv = ["maze_gen_fast", "--puzzle-file", path.to_str().unwrap()];

        let parsed = Args::try_parse_from(argv).expect("parses without --generator/--output");
        match parsed.mode().unwrap() {
            Mode::Resolve(opts) => {
                assert_eq!(opts.inputs, vec![path.clone()]);
                assert_eq!(opts.solver, SolverType::AStar);
            }
            Mode::Generate(_) => panic!("--puzzle-file must select the resolve mode"),
        }

        // Arguments that still mean something alongside it.
        for extra in [["--threads", "2"], ["--solver", "astar"]] {
            let mut with_extra = argv.to_vec();
            with_extra.extend(extra);
            assert!(Args::try_parse_from(&with_extra).is_ok(), "{extra:?}");
        }

        // Generate-only arguments are rejected rather than silently ignored.
        let rejected: [&[&str]; 14] = [
            &["--max-attempts", "5"],
            &["--list-params"],
            &["--output", "d"],
            &["--train", "5"],
            &["--test", "5"],
            &["--eval", "5"],
            &["--generator", "wilson"],
            &["--balanced"],
            &["--exclude", "d"],
            &["--param", "a=b"],
            &["--legacy-query-first"],
            &["--allow-underfill"],
            &["--fixed-maze"],
            &["--hide-walls"],
        ];
        for extra in rejected {
            let mut with_extra = argv.to_vec();
            with_extra.extend_from_slice(extra);
            assert!(
                Args::try_parse_from(&with_extra).is_err(),
                "{extra:?} should conflict with --puzzle-file"
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn puzzle_file_must_name_an_existing_file() {
        let mut a = args(None, None, Some(5));
        a.puzzle_file = Some("  ".into());
        assert!(a.mode().unwrap_err().contains("--puzzle-file"));
        a.puzzle_file = Some("/nonexistent/nope.jsonl".into());
        assert!(a.mode().unwrap_err().contains("no such file"));
        a.puzzle_file = Some(std::env::temp_dir().to_string_lossy().into_owned());
        assert!(a.mode().unwrap_err().contains("directory"));
    }

    #[test]
    fn attempt_budget_scales_with_bins() {
        let plain = args(None, None, Some(100)).validate().unwrap();
        assert_eq!(plain.max_attempts, 100 * ATTEMPTS_PER_MAZE);

        let mut a = args(None, None, Some(100));
        a.balanced = true;
        a.num_bins = 4;
        assert_eq!(
            a.validate().unwrap().max_attempts,
            100 * 4 * ATTEMPTS_PER_MAZE
        );

        // An explicit budget wins.
        a.max_attempts = Some(7);
        assert_eq!(a.validate().unwrap().max_attempts, 7);
    }
}
