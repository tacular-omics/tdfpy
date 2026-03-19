# Mass Spec Agents System — Implementation Plan

## Overview

A Claude Agent SDK-powered system for end-to-end mass spectrometry proteomics analysis, built on top of tdfpy. Agents orchestrate: raw data → mzML conversion → Sage database search → LFQ → downstream statistical analysis, with experiment tagging and metadata throughout.

## Architecture

```
src/tdfpy/agents/
├── __init__.py              # Agent registry & exports
├── cli.py                   # CLI entry point (typer)
├── config.py                # Agent config / experiment DB schema
├── models.py                # Pydantic models for experiments, metadata, tags
├── db.py                    # SQLite experiment/metadata store
├── tools/
│   ├── __init__.py
│   ├── tdf_tools.py         # Tools wrapping tdfpy: read frames, centroid, extract metadata
│   ├── conversion_tools.py  # .d → mzML conversion (timsconvert or msconvert)
│   ├── sage_tools.py        # Generate Sage JSON config, run Sage, parse results
│   ├── quant_tools.py       # LFQ result parsing, normalization, rollup
│   ├── metadata_tools.py    # Tag experiments, add metadata, query experiment DB
│   └── analysis_tools.py    # Downstream: differential expression, volcano, enrichment
├── agents/
│   ├── __init__.py
│   ├── orchestrator.py      # Top-level agent: routes user requests to sub-agents
│   ├── search_agent.py      # Runs Sage search pipeline (convert → search → FDR)
│   ├── quant_agent.py       # LFQ analysis agent (normalize, impute, rollup)
│   ├── metadata_agent.py    # Experiment tagging & metadata management
│   └── analysis_agent.py    # Downstream stats (DE, enrichment, visualization)
└── prompts/
    ├── orchestrator.md      # System prompt for orchestrator
    ├── search.md            # System prompt for search agent
    ├── quant.md             # System prompt for quant agent
    ├── metadata.md          # System prompt for metadata agent
    └── analysis.md          # System prompt for analysis agent
```

## Components

### 1. Experiment Database & Metadata (`db.py`, `models.py`)

SQLite database tracking experiments, runs, and results:

```
experiments
├── id (UUID)
├── name
├── description
├── organism
├── instrument
├── acquisition_mode (DDA/DIA/PRM)
├── created_at
└── tags (JSON array)

runs
├── id (UUID)
├── experiment_id (FK)
├── raw_path (.d folder)
├── mzml_path (converted)
├── fasta_path
├── sage_config_path
├── sage_results_path
├── status (pending/converting/searching/quantifying/complete/failed)
├── metadata (JSON - flexible key/value)
└── created_at

tags
├── id
├── name
├── category (e.g. "condition", "replicate", "treatment", "timepoint")
└── color (for visualization)

run_tags (junction)
├── run_id (FK)
└── tag_id (FK)

results
├── id
├── run_id (FK)
├── result_type (psm/peptide/protein/lfq)
├── file_path
├── summary_stats (JSON - PSM count, protein count, FDR stats)
└── created_at
```

Pydantic models for validation + serialization.

### 2. Tools (MCP-compatible functions for agents)

**tdf_tools.py** — Wraps existing tdfpy API:
- `read_experiment(path)` → metadata summary (frames, precursors, RT range, etc.)
- `get_acquisition_info(path)` → DDA/DIA/PRM + instrument details
- `extract_spectra(path, frame_ids, centroid=True)` → centroided spectra
- `get_tic(path)` → total ion chromatogram

**conversion_tools.py** — Raw → mzML:
- `convert_to_mzml(d_path, output_dir, tool="timsconvert")` → mzML path
- `validate_mzml(mzml_path)` → bool + basic stats

**sage_tools.py** — Sage search engine integration:
- `generate_sage_config(fasta, mzml_paths, params)` → JSON config path
- `run_sage(config_path, output_dir)` → results directory
- `parse_sage_results(results_dir)` → DataFrame summary (PSMs, peptides, proteins)
- `get_fdr_summary(results_dir)` → FDR stats at PSM/peptide/protein level

**quant_tools.py** — Label-free quantification:
- `parse_lfq_results(sage_results_dir)` → protein × sample intensity matrix
- `normalize_intensities(matrix, method="median")` → normalized matrix
- `impute_missing(matrix, method="min")` → imputed matrix
- `rollup_to_protein(peptide_matrix)` → protein-level rollup

**metadata_tools.py** — Experiment management:
- `create_experiment(name, organism, tags)` → experiment record
- `tag_run(run_id, tags)` → updated run
- `add_metadata(run_id, key, value)` → updated run
- `query_experiments(filters)` → matching experiments
- `get_experiment_summary(experiment_id)` → full summary with stats

**analysis_tools.py** — Downstream analysis:
- `differential_expression(matrix, group_a, group_b, method="ttest")` → DE results
- `volcano_plot(de_results, fc_cutoff, pval_cutoff)` → plot path
- `enrichment_analysis(protein_list, database="GO")` → enrichment results
- `pca_plot(matrix, labels)` → plot path
- `correlation_heatmap(matrix)` → plot path
- `export_results(experiment_id, format="tsv")` → exported file paths

### 3. Agents (Claude Agent SDK)

**Orchestrator** (`orchestrator.py`):
- Top-level agent that interprets user intent
- Routes to specialized sub-agents
- Maintains experiment context across interactions
- Has access to ALL tools

**Search Agent** (`search_agent.py`):
- Pipeline: validate input → convert .d → mzML → generate Sage config → run Sage → parse results → store in experiment DB
- Tools: tdf_tools, conversion_tools, sage_tools, metadata_tools
- Knows proteomics search parameters (tolerances, modifications, enzyme rules)

**Quant Agent** (`quant_agent.py`):
- Pipeline: parse Sage LFQ → normalize → impute → rollup → store
- Tools: quant_tools, metadata_tools
- Understands normalization strategies, batch effects, missing value patterns

**Metadata Agent** (`metadata_agent.py`):
- Manages experiment tagging, annotation, and querying
- Tools: metadata_tools, tdf_tools (for auto-detecting metadata from raw files)
- Can auto-tag based on file naming conventions or instrument metadata

**Analysis Agent** (`analysis_agent.py`):
- Pipeline: load quantified data → stats → visualization → export
- Tools: analysis_tools, quant_tools, metadata_tools
- Understands experimental design, multiple testing correction, enrichment

### 4. CLI Entry Point (`cli.py`)

```bash
# Interactive agent mode
tdfpy-agents chat

# Direct commands
tdfpy-agents search --fasta human.fasta --data /path/to/*.d
tdfpy-agents tag --experiment EXP001 --add "condition:treatment"
tdfpy-agents analyze --experiment EXP001 --compare control vs treatment
tdfpy-agents status --experiment EXP001
```

### 5. Agent System Prompts (`prompts/`)

Each agent gets a specialized system prompt with:
- Domain knowledge (proteomics, mass spec, statistics)
- Available tools and when to use them
- Decision-making guidelines (e.g., search parameter selection)
- Error handling and recovery strategies

## Implementation Order

1. **Models & DB** — Pydantic models, SQLite schema, experiment CRUD
2. **Metadata tools & agent** — Experiment creation, tagging, querying
3. **TDF tools** — Wrap existing tdfpy for agent use
4. **Conversion tools** — .d → mzML pipeline
5. **Sage tools & search agent** — Config generation, search execution, result parsing
6. **Quant tools & agent** — LFQ parsing, normalization, rollup
7. **Analysis tools & agent** — DE, enrichment, visualization
8. **Orchestrator** — Top-level routing agent
9. **CLI** — typer-based entry point
10. **Tests** — Unit tests for tools, integration tests for agents

## Dependencies to Add

```toml
[project.optional-dependencies]
agents = [
    "claude-agent-sdk",
    "typer",
    "pydantic>=2.0",
    "scipy",           # stats for DE analysis
    "matplotlib",      # plotting
    "seaborn",         # enhanced plots
    "scikit-learn",    # PCA, clustering
    "pyteomics",       # mzML parsing/validation
]
```

## Key Design Decisions

1. **sagepy vs sage CLI**: Use Sage CLI binary (not sagepy) for search — it's faster, more stable, and handles LFQ natively. sagepy is useful for programmatic result access but the CLI is the production path.

2. **SQLite for experiment DB**: Lightweight, no server, portable. Same approach tdfpy already uses for .tdf metadata.

3. **Claude Agent SDK**: Agents use the SDK's tool system. Each tool is a Python function decorated for MCP compatibility. Agents communicate through the experiment DB, not direct function calls.

4. **Conversion**: timsconvert (Python, timsTOF-native) preferred over msconvert for .d files. Falls back to msconvert if available.

5. **Separation of concerns**: Each agent owns a specific domain. The orchestrator delegates but doesn't do the work itself.
