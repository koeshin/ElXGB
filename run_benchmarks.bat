@echo off
setlocal enabledelayedexpansion

:: ELXGB Benchmark Automation (Windows Batch)
:: Usage: run_benchmarks.bat

set TREES=1 5 10 15 20
set DEPTHS=2 3 4 5 6
set BINS=4 6 8 12
set DATASETS=bank_marketing credit_card

for %%d in (%DATASETS%) do (
    for %%t in (%TREES%) do (
        for %%e in (%DEPTHS%) do (
            for %%b in (%BINS%) do (
                echo ========================================================================
                echo Running: Dataset=%%d, Trees=%%t, Depth=%%e, Bins=%%b
                echo ========================================================================
                python benchmark/benchmark_runner.py --dataset %%d --trees %%t --depth %%e --bins %%b
            )
        )
    )
)

echo ✅ All benchmarks completed!
pause
