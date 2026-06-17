# AI and Youth Employment: Causal Inference Analysis

Final Project for *Causal Inference and AI*

## Research Question

**Does the advancement of generative AI causally increase youth employment anxiety?**

This project investigates whether the rapid development and adoption of generative AI has a causal effect on youth employment uncertainty using Judea Pearl's causal inference framework and a custom-built news article corpus.

---

## Repository Structure

```text
.
├── articles/               # Original news articles
├── cleaned/                # Cleaned and normalized articles
├── extractions/
│   ├── entities.csv        # Extracted entities
│   └── causal_assertions.csv # Extracted causal claims
├── knowledge_graph/
│   ├── nodes.csv
│   ├── edges.csv
│   ├── source_nodes.csv
│   └── graph.png
├── causal_dag/
│   ├── dag.png
│   └── dag_edges.csv
├── metadata.csv            # Article metadata
├── step2_clean.py          # Text cleaning pipeline
├── run_pipeline.py         # Knowledge graph and DAG generation
├── analysis_report.md      # Final analysis report
└── README.md
