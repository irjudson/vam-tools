#!/bin/bash
# Run all tests in parallel chunks for maximum speed

set -e

echo "🧹 Cleaning cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
rm -rf .pytest_cache .coverage htmlcov 2>/dev/null || true

echo "📝 Collecting test files..."
mapfile -t test_files < <(find tests/ -name "test_*.py" -type f | sort)

total_files=${#test_files[@]}
echo "📊 Found $total_files test files"

# Create output directory
mkdir -p /tmp/test_results

echo ""
echo "🚀 Running tests in parallel (8 at a time)..."
echo "⏱️  Note: Full test suite may take 5-10 minutes due to image processing tests"
echo "   Fast tests (DB, API): <1 second each"
echo "   Slow tests (analysis, scanning): 10-60 seconds each"
echo ""
start_time=$(date +%s)

# Function to run a single test file
run_test() {
    local test_file=$1
    local index=$2
    local total=$3
    local output_file="/tmp/test_results/$(basename "$test_file" .py).txt"

    local test_start=$(date +%s)
    echo "[$index/$total] Running $test_file..."
    # Use -p no:xdist to disable xdist parallelization within each file
    # (we're already parallelizing at the file level)
    # Use -o addopts="" to ignore pyproject.toml settings
    ./venv/bin/pytest "$test_file" -q --tb=line -p no:warnings -p no:xdist -o addopts="" 2>&1 > "$output_file"
    local exit_code=$?
    local test_end=$(date +%s)
    local test_duration=$((test_end - test_start))

    if [ $exit_code -eq 0 ]; then
        echo "✓ [$index/$total] PASSED: $test_file (${test_duration}s)"
    else
        echo "✗ [$index/$total] FAILED: $test_file (exit code: $exit_code, ${test_duration}s)"
    fi

    return $exit_code
}

export -f run_test

# Track running jobs
running=0
max_parallel=8

# Run tests with manual parallelization
for i in "${!test_files[@]}"; do
    test_file="${test_files[$i]}"
    index=$((i + 1))

    # Run test in background
    run_test "$test_file" "$index" "$total_files" &

    ((running++))

    # Wait if we've hit the parallel limit
    if [ $running -ge $max_parallel ]; then
        wait -n  # Wait for any one job to finish
        ((running--))
    fi
done

# Wait for all remaining jobs
wait

end_time=$(date +%s)
duration=$((end_time - start_time))

echo ""
echo "⏱️  Total time: ${duration}s ($(($duration / 60))m $(($duration % 60))s)"
echo ""
echo "📋 Collecting results..."

# Collect and display summary
passed=0
failed=0
error=0

for test_file in "${test_files[@]}"; do
    output_file="/tmp/test_results/$(basename "$test_file" .py).txt"
    if [ -f "$output_file" ]; then
        if grep -q "passed" "$output_file" && ! grep -q "failed\|ERROR" "$output_file"; then
            ((passed++))
        else
            ((failed++))
            echo "FAILED: $test_file"
            tail -5 "$output_file" | sed 's/^/  /'
        fi
    else
        ((error++))
        echo "ERROR: No output for $test_file"
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════"
echo "📊 FINAL SUMMARY"
echo "═══════════════════════════════════════════════════════"
echo "✓ Passed: $passed"
echo "✗ Failed: $failed"
echo "⚠ Error:  $error"
echo "📁 Total test files: $total_files"
echo "⏱  Total duration: ${duration}s ($(($duration / 60))m $(($duration % 60))s)"
echo "⚡ Average per file: $((duration / total_files))s"
echo ""
echo "Performance breakdown:"
echo "  - Session setup (DB schema): ~1s (once for all tests)"
echo "  - Fast tests (DB, API, utils): <1s each"
echo "  - Medium tests (analysis): 5-15s each"
echo "  - Slow tests (scanning, duplicates): 20-60s each"
echo "═══════════════════════════════════════════════════════"

if [ $failed -gt 0 ] || [ $error -gt 0 ]; then
    exit 1
fi
