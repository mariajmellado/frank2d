# Development guide for Frankenstein 2D 🌈

This document describes the branch and commit conventions adopted for this project, to keep the history organized and make it easier to work in parallel on CPU and GPU.

## Branch structure

These are the persistent branches of the project:

- **`main`**: stable code. Only receives merges from `develop` or `develop_gpu` when a set of changes is ready to be versioned. Never work directly here.
- **`develop`**: integration branch for CPU-oriented improvements. This is where CPU-related sub-branches are merged before moving to `main`.
- **`develop_gpu`**: integration branch for GPU-oriented improvements. Same role as `develop`, but for the GPU side of the project.

## Sub-branch naming convention

Every other branch is a temporary sub-branch created off `develop` or `develop_gpu` (never off `main`), named with one of the following prefixes:

| Prefix | Use |
|---|---|
| `feat/` | New functionality |
| `fix/` | Bug fix |
| `refactor/` | Code restructuring without changing behavior |
| `docs/` | Documentation changes |
| `test/` | Adding or modifying tests |
| `chore/` | Maintenance tasks (dependencies, configuration, etc.) |

Examples already used in the repo: `feat/fit_error`, `feat/new_griddings`, `feat/optimization`, `feat/real-fft`, `fix/brightness-difference-with-frank1d`.

## Commit convention

Format: `type: short description in present tense`

- `feat:` new functionality for the user
- `fix:` bug fix
- `docs:` documentation only
- `refactor:` code change that neither fixes a bug nor adds a feature
- `test:` adding or fixing tests
- `perf:` change focused on improving performance
- `chore:` tasks that don't touch production code (configs, dependencies)

Examples:

```
feat: add adaptive gridding for sparse uv-coverage
fix: correct residual sign in Fourier plot
refactor: extract FWHM calculation into a separate function
docs: document fit_error parameters in README
```

If a commit needs more context, add a body after a blank line:

```
fix: correct Im(V) cancellation in FFT grid

The bug was in get_profile, which used f2d.u_grid/v_grid instead
of the actual observed baselines.
```

## Workflow for a new feature

1. Update the base branch before starting:
   ```bash
   git checkout develop
   git pull
   ```
2. Create the feature branch from `develop` (or `develop_gpu` if it's GPU-related):
   ```bash
   git checkout -b feat/descriptive-name
   ```
3. Work with small commits following the convention above.
4. **If the feature is large** and it makes sense to split it into parts, create sub-branches from the feature branch itself (not directly from `develop`):
   ```bash
   git checkout feat/descriptive-name
   git checkout -b feat/descriptive-name/subpart-1
   ```
   Each sub-branch gets merged back into `feat/descriptive-name` as it's ready. Once the whole feature is finished, `feat/descriptive-name` is then merged into `develop`.
5. Merge into `develop`:
   ```bash
   git checkout develop
   git merge feat/descriptive-name
   ```

## How changes move to main (versioning)

When `develop` (or `develop_gpu`) has accumulated a stable, tested set of changes:

```bash
git checkout main
git pull
git merge develop
git tag -a v1.3.0 -m "Short description of the version"
git push origin main --tags
```

`main` must always stay in a working state. Never merge to `main` with half-tested code.

## Keeping develop and develop_gpu up to date

If `main` receives changes (for example a hotfix) that also need to be available in both development branches:

```bash
git checkout develop
git merge main

git checkout develop_gpu
git merge main
```

## General best practices

- Small, frequent commits, each with a clear purpose.
- Never push directly to `main`.
- Test changes before merging into `develop` or `develop_gpu`.
- If a branch was already pushed and gets corrected with `--amend` or `rebase`, use `git push --force-with-lease`, never plain `--force`.
- Delete `feat/` or `fix/` branches once merged, to keep `git branch` clean.