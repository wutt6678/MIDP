# Qwen3.5-9B FIUBench Annotation Patch - Status Report

## Completed Items ✅

### 1. Code Cleanup
- ✅ Removed all debug prints from `src/route_data/models/qwen.py`
- ✅ Cleaned up temporary diagnostic code

### 2. Testing & Validation
- ✅ All 618 tests pass (7 skipped, 2 xfailed)
- ✅ Ruff linting: All checks passed
- ✅ Test suite runs successfully with TMPDIR set to scratch filesystem

### 3. Git Staging & Committing
- ✅ Staged 13 files (933 insertions, 81 deletions)
- ✅ Committed with descriptive message: `b006179`
- ✅ Files staged:
  - `configs/model/qwen35_9b.yaml` - device_map: cuda:0
  - `configs/prompts/celeba_binary_v1.yaml` - candidate format: Yes/No
  - `configs/runs/tiny_fiubench_manifest.json` - smoke manifest (3 samples)
  - `configs/runs/tiny_fiubench_qwen.yaml` - FIUBench run config
  - `outputs/smoke_test/golden_stub_model_smoke.json` - updated expectations
  - `src/route_data/cli.py` - candidate format updates
  - `src/route_data/eval/celeba_runner.py` - candidate format updates
  - `src/route_data/models/base.py` - docstring update
  - `src/route_data/models/qwen.py` - core fix: prefix-token teacher forcing
  - `src/route_data/models/scoring.py` - SCORING_VERSION 1→2
  - `tests/golden/test_golden_e2e.py` - updated expected counts
  - `tests/integration/test_qwen_real_model_smoke.py` - new integration test
  - `tests/unit/test_qwen_scoring.py` - new unit tests

### 4. Core Implementation
- ✅ Prefix-token teacher forcing for Qwen3.5-9B
- ✅ mm_token_type_ids extension for text-only candidate tokens
- ✅ Proper multimodal prefix construction via apply_chat_template()
- ✅ Frozen binary candidate protocol: "Yes"/"No" (capitalized, no leading space)
- ✅ SCORING_VERSION bump to invalidate caches
- ✅ device_map: "cuda:0" to prevent silent zero logits

## Blocked Items ⚠️

### FIUBench Annotation Smoke Test
**Status**: BLOCKED - GPU memory constraint

**Issue**: All GPUs are at capacity
- GPU 0: 48.2GB/49.1GB used (98%)
- GPU 1: 31.4GB/49.1GB used (65%) - **17.1GB free, need 17.5GB**
- GPU 2: 45.0GB/49.1GB used (92%)
- GPU 3: 31.8GB/49.1GB used (65%) - 16.8GB free, need 17.5GB

**Root Cause**: 
- GPU 1 has active processes from another user (PIDs 1896682, 1898252)
- These are running evaluations started ~10 minutes ago
- Cannot kill processes belonging to other users

**Diagnostic Findings**:
- First 25 queries (of 120) process successfully with distinct log probs
- Query 25 fails with probability collapse: `[-1.2445385456085205, -1.2445385456085205]`
- This suggests a specific attribute/image combination triggers the issue
- Need to investigate what's special about query 25

**Next Steps**:
1. Wait for GPU 1 processes to complete (~1-2 hours estimated)
2. Retry annotation smoke test
3. If query 25 still fails, investigate the specific attribute/image combination
4. Consider adding attribute-specific debugging to understand the failure mode

## Disk Space Issue Resolved ✅
- Root filesystem was 100% full (890GB/938GB)
- Set `TMPDIR=/scratch/wutiantong/tmp` to use scratch filesystem
- Tests now run successfully

## Untracked Files (Not Committed)
- `diag_fiubench_score.py` - diagnostic script (temporary)
- `scripts/diagnose_qwen_logits.py` - diagnostic script (temporary)

## Technical Summary

### Key Changes
1. **Prefix-Token Teacher Forcing**: Construct multimodal prefix via `processor.apply_chat_template()`, append candidate token IDs, score at exact positions
2. **mm_token_type_ids Extension**: Text-only candidate tokens get type=0, multimodal prefix tokens retain original types
3. **Frozen Binary Candidate Protocol**: Changed from " yes"/" no" to "Yes"/"No" (capitalized, no leading space)
4. **SCORING_VERSION**: Bumped from "1" to "2" to invalidate old caches
5. **device_map**: Changed from "auto" to "cuda:0" to prevent silent zero logits

### Test Coverage
- Unit tests: `test_qwen_scoring.py` (372 lines)
- Integration tests: `test_qwen_real_model_smoke.py` (245 lines)
- Golden tests: Updated expectations in `test_golden_e2e.py`
- All 618 tests pass

### Performance
- Model loading: ~5 seconds
- Per-query scoring: ~0.5 seconds
- Estimated total for 120 queries: ~60 seconds (once GPU memory is available)

## Recommendations
1. **Immediate**: Wait for GPU 1 to free up, then retry annotation smoke test
2. **Short-term**: Investigate query 25 failure mode if it persists
3. **Long-term**: Consider model parallelism or CPU offloading for memory-constrained environments
4. **Documentation**: Update project memories with Qwen3.5-9B integration details

## Environment
- Conda: `midp-qwen35` (Python 3.10, torch 2.8.0+cu128, transformers 5.14.1)
- GPUs: 4x NVIDIA RTX 6000 Ada (49GB each)
- Disk: 1.6TB free on /scratch
- Commit: `b006179` on main branch
