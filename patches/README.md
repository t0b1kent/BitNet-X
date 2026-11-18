# patches/

Fixes to the pinned `3rdparty/llama.cpp` submodule
(commit `390c307752ab78fd8189f359d6954c9ba1be74af`).

The submodule pin is never changed and nothing is committed inside the
submodule. Each fix is one patch file `patches/llama.cpp/NNNN-<topic>.patch`
(plain `git diff` relative to the submodule root), applied to the submodule
working tree by `scripts/apply_patches.sh`:

```bash
scripts/apply_patches.sh            # apply (idempotent)
scripts/apply_patches.sh --check    # report, exit 1 if something is missing
scripts/apply_patches.sh --reverse  # restore the pristine submodule
```

`cmake` runs `scripts/apply_patches.sh` at configure time
(`-DBITNETX_APPLY_PATCHES=OFF` disables that).

Rules:

* one independent defect = one patch = one pull request;
* patches must apply independently of each other (different files or hunks);
* each patch has a test in `tests/` that fails without it and passes with it.

To create a patch: edit the submodule working tree, then

```bash
git -C 3rdparty/llama.cpp diff -- <files> > patches/llama.cpp/NNNN-topic.patch
```
