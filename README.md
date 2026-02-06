# YoseTsumego

This project generates Go problems using KataGo.

## Project Structure

The project code is organized into three main directories:
- **`pipeline/`**: Executable scripts that form the core workflow (generation, benchmarking, ranking, packaging).
- **`utils/`**: Core logic, shared helpers, and engine controllers.
- **`auxiliary/`**: Analysis tools, diagnostic scripts, and legacy code.

## Pipeline Execution

To generate, rank, and package tsumegos, run the following scripts in order using the virtual environment:

### 1. Fetch Database
Downloads the latest tsumego attempt data from Cloudflare D1.
```bash
./.venv/bin/python3 -m pipeline.fetch_database
```

### 2. Tune Engine
Optimizes KataGo batch sizes for your hardware.
```bash
./.venv/bin/python3 -m pipeline.tune_engine
```

### 3. Generate Tsumegos
Continuously generates new tsumego positions.
```bash
./.venv/bin/python3 -m pipeline.generate_tsumegos
```

### 4. Benchmark Bots
Runs various bots against the generated tsumegos to collect performance data.
```bash
./.venv/bin/python3 -m pipeline.benchmark_bots
```

### 5. Rank Tsumegos and Bots
Trains the rating model based on the benchmark results.
```bash
./.venv/bin/python3 -m pipeline.rank_tsumegos_and_bots
```

### 6. Package for PWA
Groups tsumegos and ratings into ZIP files for the web app.
```bash
./.venv/bin/python3 -m pipeline.package_for_pwa
```

## PWA

This project includes a Progressive Web App (PWA) for solving tsumegos.

### Building the PWA

To build the PWA, navigate to the `pwa` directory and run the build command:

```bash
cd pwa
npm install
npm run build
```

### Problem Selection Logic

The PWA uses a history-based problem selection logic:

- It tracks the last 200 problems you've faced.
- Each problem is marked as `success`, `failure`, or `in-between`.
- Failed problems have a linearly increasing probability of being shown again.
- In-between problems also have a (slower) linearly increasing probability of being shown again.
- If no problem is selected from the history, a new problem close to your rating is chosen.
