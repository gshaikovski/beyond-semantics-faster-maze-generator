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
