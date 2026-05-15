# pycellchat

A Rust-powered Python library for **cell-cell communication inference** in single-cell and spatial transcriptomics data. Native AnnData/Scanpy compatibility with high-performance computation.

## Features

- **Rust-powered core**: Heavy computations (Hill function, permutation testing, SNN construction, centrality metrics) implemented in Rust via PyO3
- **Native AnnData support**: Seamless integration with Scanpy ecosystem
- **Multi-species databases**: Built-in interaction databases for human, mouse, and zebrafish
- **Spatial analysis**: Support for spatial transcriptomics data with distance-based communication modeling
- **Comprehensive visualization**: Heatmaps, bubble plots, chord diagrams, hierarchy plots, and more

## Installation

### Prerequisites

- Python >= 3.10
- Rust toolchain (install via [rustup](https://rustup.rs/))

### Install from source

```bash
git clone https://github.com/yourusername/pycellchat.git
cd pycellchat
pip install maturin
maturin develop --release
```

### Dependencies

- anndata >= 0.10
- scanpy >= 1.9
- numpy >= 1.23
- scipy >= 1.10
- pandas >= 1.5
- matplotlib >= 3.6
- seaborn >= 0.12
- scikit-learn >= 1.2

## Quick Start

```python
import pycellchat
import scanpy as sc

# Load your data
adata = sc.read_h5ad("your_data.h5ad")

# Initialize CellChat object
cc = pycellchat.CellChat(adata, group_by="cell_type")

# Set interaction database
cc.set_db("human")  # or "mouse", "zebrafish"

# Run analysis pipeline
cc.normalize()
cc.subset_data()

from pycellchat.modeling import compute_commun_prob, CommunProbParams
params = CommunProbParams(nboot=100, seed=42)
compute_commun_prob(cc, params)

# Visualize results
from pycellchat.plotting import net_visual_heatmap
fig = net_visual_heatmap(cc)
```

## Project Structure

```
pycellchat/
├── crates/
│   ├── pycellchat-core/        # Pure Rust computation engine
│   └── pycellchat-py/          # PyO3 bindings (Rust -> Python)
├── python/
│   └── pycellchat/             # Python package
│       ├── database/           # Interaction databases (Parquet format)
│       └── plotting/           # Visualization functions
├── tests/                      # Test suite
└── scripts/                    # Utility scripts
```

## Testing

```bash
# Install dev dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=pycellchat tests/
```

## API Reference

### Core Classes

- `CellChat`: Main class for cell-cell communication analysis
- `CommunProbParams`: Parameters for communication probability computation

### Key Functions

- `compute_commun_prob()`: Compute communication probabilities
- `compute_commun_prob_pathway()`: Aggregate probabilities by pathway
- `aggregate_net()`: Aggregate network statistics
- `net_analysis_compute_centrality()`: Compute network centrality metrics
- `identify_communication_patterns()`: Identify communication patterns (NMF)

### Visualization

- `net_visual_heatmap()`: Heatmap of communication probabilities
- `net_visual_bubble()`: Bubble plot of signaling pathways
- `net_visual_chord_cell()`: Chord diagram of cell communications
- `rank_net_plot()`: Rank plot of signaling pathways
- `net_visual_hierarchy()`: Hierarchy visualization

## Database

The package includes interaction databases for three species:

- **Human**: 3,233 interactions
- **Mouse**: 3,379 interactions
- **Zebrafish**: 2,774 interactions

Databases are stored in Parquet format in `python/pycellchat/database/data/`.

## Performance

The Rust core provides significant speedups for computationally intensive operations:

- Hill function computation
- Permutation testing (p-value calculation)
- SNN graph construction
- Centrality metrics computation

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use pycellchat in your research, please cite:

```bibtex
@software{pycellchat,
  author = {Your Name},
  title = {pycellchat: Rust-powered cell-cell communication inference},
  year = {2025},
  url = {https://github.com/yourusername/pycellchat}
}
```
