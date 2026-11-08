# Beyond Semantics -- Faster Data Generation Utilities

See [here](https://github.com/kstechly/beyond-semantics-maze-visualizer) for background. This is just a faster reimplementation of the data generation pipeline (though it does not replicate bit-for-bit the same datasets as in [the paper](https://arxiv.org/abs/2505.13775)).

## Prerequisites
- Rust 1.70 or later

## Basic Usage

One invocation writes a whole **dataset directory**: the requested partition files plus `args.json` (the parsed CLI args, for provenance).

```bash
# 5M train + 1000 test 20x20 Wilson mazes, uniform over plan length
./target/release/maze_gen_fast -g wilson -s astar \
    --rows 20 --cols 20 --seed 0 \
    --train 5000000 --test 1000 \
    --balanced --num-bins 100 \
    -o /data/datasets/maze_wilson_20_5000000_seed_0_balanced
```

```
maze_wilson_20_5000000_seed_0_balanced/
├── args.json      # the invocation that wrote this directory
├── test.jsonl
└── train.jsonl
```

`--eval`, `--test` and `--train` are independent; pass any subset (at least one). An existing partition file is never overwritten — the run aborts instead.

Use different generators:
```bash
# Kruskal's algorithm
./target/release/maze_gen_fast -g kruskal -s astar --train 1000000 --seed 37 -o mazes_kruskal

# Searchformer style (random walls validated by pathfinding)
./target/release/maze_gen_fast -g searchformer -s astar --train 1000000 --seed 999 -o mazes_sf

# Drunkard's walk with custom coverage
./target/release/maze_gen_fast -g drunkards-walk -s astar --train 1000000 \
    --param coverage=0.7 --seed 666 -o mazes_dw
```

List generator parameters:
```bash
# List parameters for all generators
./target/release/maze_gen_fast --list-params

# List parameters for a specific generator
./target/release/maze_gen_fast --list-params drunkards-walk
```

## Split disjointness

Partitions are generated in the fixed order **eval → test → train** from a single stream of instance ids, sharing one seen-set of emitted **wall layouts**. A layout claimed by an earlier split can never be claimed again, so the splits are disjoint by construction rather than by hoping two master seeds diverge.

Identity is the layout alone — grid shape plus the wall bitmap — ignoring start and goal: two instances with the same walls share the same A\* search graph and differ only in where the search enters and leaves, so re-emitting one would leak a solved layout into eval. (Keys are 128-bit FNV-1a digests, not stored layouts.)

Consequences worth knowing:
- A run that cannot find enough distinct layouts **fails loudly** with a per-bin deficit table rather than silently emitting duplicates. Pass `--allow-underfill` to accept a short dataset, or raise `--max-attempts`.
- `--balanced` bins each partition separately, so every split is uniform over plan length on its own.

### Staying disjoint from an existing dataset

`--exclude <dataset dir>` scans that directory's partition files and pre-seeds the seen-set with every layout in them:

```bash
# an OOD eval set guaranteed to share no layout with the training dataset
./target/release/maze_gen_fast -g wilson -s astar --rows 20 --cols 20 --seed 7 \
    --eval 1000 -o maze_wilson_20_ood_eval \
    --exclude /data/datasets/maze_wilson_20_5000000_seed_0_balanced
```

Records whose grid shape differs from the current `--rows`/`--cols` are skipped (they cannot collide); an unparsable line is a hard error, since silently ignoring one would leave the datasets overlapping.

## One maze for every instance (`--fixed-maze`)

`--fixed-maze` draws a single wall layout and uses it for every instance in every split; each instance draws its own start and goal, so only the query, the trace and the plan vary.

```bash
./target/release/maze_gen_fast -g wilson -s astar --rows 20 --cols 20 --seed 0 \
    --train 30000 --test 1000 --fixed-maze -o maze_wilson_20_fixed_seed_0
# Fixed maze: 199 floor cells, 39402 distinct queries
```

- The layout is a function of `--seed`/`--generator`/`--rows`/`--cols` alone (its own reserved PRNG stream), so two runs with the same arguments share it.
- Start/goal are uniform over **ordered** pairs of distinct floor cells. Identity becomes the query instead of the layout: no `(start, goal)` pair repeats within or across splits. `(s, g)` and `(g, s)` are different instances; their plans are reverses of each other.
- Capacity is `n(n-1)` for `n` floor cells, printed at start; a larger request fails before anything is written. `--balanced` is further limited by how few pairs have long paths on the one layout.
- Conflicts with `--exclude` (layout exclusion is meaningless when every instance shares the layout).
- `searchformer` layouts are not connected: unreachable pairs are skipped, and its usual minimum path length (`>= max(rows, cols)`) is not enforced per query.

## Re-solving an existing file (`--puzzle-file`)

`--puzzle-file <path.jsonl>` reads mazes instead of generating them: each record is re-solved
with this crate's A\* and written to `<dir>/<stem>_resolved.jsonl` beside the input.

```bash
./target/release/maze_gen_fast --puzzle-file /data/datasets/maze_wilson_20/test_looped.jsonl
# -> /data/datasets/maze_wilson_20/test_looped_resolved.jsonl
```

**Everything except the trace and the plan is copied verbatim** — every JSON key, its value and
its position, including keys this tool knows nothing about, and inside `text` every byte before
` reasoning`. Only the search trace and the plan are replaced. The line is spliced rather than
re-encoded, so key order survives and the record needs no round-trip through a JSON parser.

The motivating case is a file whose stored trace has gone stale — e.g. `test_looped.jsonl`,
where a wall was deleted from `text` but the trace and plan still describe the route from before
the deletion. The run reports what moved:

```
Resolved 740 records into …/test_looped_resolved.jsonl in 0.01s
  0/740 unchanged (the stored trace and plan already match this A*)
  plan:  shorter 740, longer 0 (mean -19.7 steps over the 740 changed)
  trace: shorter 740, longer 0 (mean -104.8 events over the 740 changed)
```

Because A\* is deterministic, re-solving an unmodified file is a no-op: the output is
byte-identical to the input, and the summary says so.

Worth knowing:

- **Nothing else is written.** No `args.json`, no sidecar — a source dataset's provenance is
  never touched. The output is built under a temporary name and renamed into place only on
  success, so an aborted run leaves no partial file. An existing `_resolved.jsonl` is never
  overwritten.
- **Any record that fails aborts the whole run**, naming the file and line: no start→goal path,
  no `rows`/`cols` keys (dimensions cannot be safely inferred), a start or goal on a wall, an
  unrecognised token in the maze spec. A file is either fully re-solved or not written at all.
- Both text layouts are read (query-first and walls-first) and the input's layout is preserved,
  since the prompt bytes are never rewritten.
- The record's own `solver` field is copied, not consulted; re-solving always uses this crate's
  A\*. `seed`, `generator` and `idx` are likewise preserved and, after a re-solve, no longer
  describe the trace in the record.
- Records must be compact JSON (`{"key":value}`). A file written with Python's default
  `json.dumps` spacing is rejected — the same limitation `--exclude` has.
- `--puzzle-file` conflicts with the generation arguments (`--generator`, `--output`,
  `--train/--test/--eval`, `--balanced`, `--exclude`, …) rather than ignoring them.
  `--threads` still sizes the pool.

## Determinism

Candidates are generated in parallel but accepted serially in ascending instance-id order, so output is byte-identical for the same arguments at any `--threads`. `idx` keeps the generating instance id, which stays ascending (and sparse when candidates are rejected).

### Output Format
The tool outputs mazes in JSONL format (one JSON object per line). This should be compatible with the main pipeline, though it does contain additional fields. Each line contains:
- `idx`: Instance ID
- `text`: Maze representation with start/goal positions, walls, reasoning trace, and solution path
- `generator`: Algorithm used to generate the maze
- `solver`: Algorithm used to solve the maze
- `seed`: Random seed used
- `rows`: Maze height
- `cols`: Maze width

#### Where the query block sits

`text` puts the `query start ... goal ...` block **after** the wall list, immediately before `reasoning`:

```
wall 1 0 wall 0 2 query start 0 0 goal 2 2 reasoning close ... solution plan ... end
```

The query then sits next to the trace it conditions rather than being separated from it by the whole maze body, which for a large grid is most of the prompt. `--legacy-query-first` restores the original layout:

```
query start 0 0 goal 2 2 wall 1 0 wall 0 2 reasoning close ... solution plan ... end
```

Only the order changes — same tokens, same wall order, same trace. `--exclude` reads either layout, and the flag is recorded in `args.json`.

`--hide-walls` drops the wall list entirely, so the prompt is the query alone (meant for `--fixed-maze`, where the layout is the same in every record):

```
query start 0 0 goal 2 2 reasoning close ... solution plan ... end
```

The layout is then not recoverable from the record: anything that reads walls back out of `text` (`--exclude`, `--puzzle-file`, a plan verifier) sees an open grid. Conflicts with `--legacy-query-first`.
