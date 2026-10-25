# AI use — working record

---

## 1. Tools used

| Tool | Used for | Period |
|---|---|---|
| Claude (Anthropic) | Code generation, test design, experiment design, analysis of results, explanation of methods | October 2026 |

No other AI tools were used. *(Add any others, and the specific interfaces/versions you used.)*

## 2. What AI was used for, by phase

| Phase | AI contribution | Output |
|---|---|---|
| Dataset selection | Proposed three candidate datasets with validation evidence; discussed feasibility | Choice of C-MAPSS FD001 |
| Project plan | Drafted a phased plan from the brief | Plan followed through the project |
| Data loading | Wrote `loader.py` and its tests | 31 tests |
| Exploratory analysis | Wrote `01_eda_fd001.ipynb`, `features.py`; proposed the trend-strength screening | 14 sensors kept; RUL cap of 125 proposed |
| Validation design | Wrote `splits.py` with the engine-grouped split and leakage guard | 21 tests |
| Metrics | Wrote `metrics.py`; identified the constant discrepancy in the source paper | 31 tests |
| Modelling | Wrote `models.py`, `windows.py`, `cv.py` and the experiment scripts | Baseline → windowed → boosting |
| Sensitivity | Designed and ran the cap and window sweeps | Window changed 30 → 50 |
| Cost-aware objective | Wrote the cost model and quantile-loss training | `decision.py`, `run_asymmetric.py` |
| Performance analysis | Wrote `profiling.py`; derived the FLOP counts; found and fixed the pipeline bottleneck | 3.1× end-to-end speedup |
| Final evaluation | Wrote `run_final_test.py` against the configuration frozen beforehand | Single test-set run |
| Documentation | Drafted this README and repository scaffolding | — |

## 3. Errors, limitations and unhelpful suggestions produced by AI

Recorded as they occurred. These are the specific items to draw on for section 7 of the report.

1. **Wrong prediction about what would improve the model.** AI identified per-engine baseline
   normalisation as "the single idea most likely to reduce the error further". Tested under the
   same protocol, it was *worse* in all 15 paired folds, for both models (boosting RMSE 14.64 →
   16.34). The reasoning sounded plausible and was wrong; only the experiment settled it.

2. **A suggestion that broke the test suite.** AI recommended saving the pytest output as
   `test_report.txt`. pytest collects files matching `test*.txt` as doctest files, and PowerShell's
   `>` redirect writes UTF-16, so the next test run failed with a `UnicodeDecodeError` during
   collection. Fixed by renaming to `pytest_report.txt` and adding `pytest.ini` with
   `testpaths = tests`.

3. **A test that encoded incorrect reasoning.** A test asserted that the arithmetic intensity of
   ridge prediction approaches 0.25 as the number of *rows* grows. It actually equals
   `p / (4(p+1))` and approaches 0.25 as the number of *features* grows. The test failed, and the
   test was wrong, not the code.

4. **A test that failed for the wrong reason.** A test comparing boosted trees with a linear model
   on a capped target scored rows whose true RUL exceeded the cap — an error no capped model can
   avoid — which masked the real difference. The test design was corrected (restricting the scored
   range) rather than its threshold loosened.

5. **An incorrect claim about the implementation.** AI reported that the windowing step
   "materialises every trailing window" and so dominated memory. In fact `sliding_window_view`
   returns a strided view: the 110 MB is never allocated, and measured peak was 15 MB.

6. **Misleading memory measurements.** Peak-memory figures initially included first-call overhead,
   reporting 60 MB for a boosting fit that actually uses 5.4 MB. Corrected by adding a warm-up call.

7. **An incorrect test count.** AI reported 211 passing tests; its internal runner's module list
   had silently omitted one test file. The true total was 230.

8. **Test counts were never produced by pytest.** pytest could not be installed in the AI's
   environment, so every count it quoted came from a substitute runner. All counts had to be
   verified by running pytest locally — which is how discrepancy 7 was found.

9. **Minor code defects.** An unused import left in `run_final_test.py`; results tables printed
   without row labels in one script.

## 4. Issues AI identified that proved correct

Recorded because they are also evidence about where AI helped, and each was independently verified.

1. **The source paper's scoring constants are inconsistent with its own text.** Equation 11 of
   Saxena et al. (2008), as printed, assigns the constants so that *early* predictions are penalised
   more heavily, contradicting the paper's stated intent. Verified against four independent
   publications, which all use 13 for early and 10 for late errors.

2. **The dataset readme has FD004's engine counts swapped.** It states 248 train / 249 test; the
   files contain 249 train / 248 test, and `RUL_FD004.txt` has 248 values. Verified by counting
   directly from the files.

3. **A leakage trap in a library default.** scikit-learn's gradient boosting enables early stopping
   automatically at this dataset size, holding out a random 10% of *rows*. Because consecutive
   cycles of one engine are near-duplicates, that check would have leaked. Disabled explicitly.

## 5. How AI-generated material was checked

- **230 automated tests**, the majority written from hand-computed values rather than recorded
  outputs, so they verify behaviour rather than pin whatever the code happened to do.
- **Mutation testing of the safeguards.** Deliberately broken implementations — a row-level split,
  a feature function that reads future cycles, a baseline that uses the whole engine, swapped score
  constants — were substituted to confirm the relevant tests fail. A test that cannot fail proves
  nothing.
- **Agreement with independent implementations.** The hand-written ridge matches scikit-learn's to
  1e-8; the vectorised trend screening matches a plain column-by-column implementation to 1e-16.
- **Verification against primary sources.** The score constants were checked against published
  papers; the engine counts against the data files; the sensor list against magnitudes in the data.
- **Independent execution.** All results were re-run locally with real pytest and real scripts,
  rather than accepted as reported.

---
