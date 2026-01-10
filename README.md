# Knowledge Graph-Enhanced Drug-Gene Interaction Prediction in Alzheimer's Disease: A GNN-LLM Approach

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🧠 Overview

This project presents a novel computational framework for drug-gene interaction analysis and hypothesis generation in Alzheimer's disease by combining **Knowledge Graphs**, **Graph Neural Networks (GNNs)**, and **Large Language Models (LLMs)**. The system predicts drug-gene interactions using the AlzKB knowledge graph and provides AI-powered explanations for the predictions.

### Novel Contributions

- **Knowledge Graph-Enhanced Predictions**: Leverages structured biomedical knowledge from AlzKB rather than relying solely on molecular features
- **Multi-Modal Architecture**: Combines graph topology, node features, and semantic embeddings for comprehensive analysis
- **Constrained LLM Explanations**: AI-generated interpretations are grounded in graph-derived evidence, reducing hallucination
- **Ensemble Learning**: RGCN + RGAT models provide robust confidence-aware predictions

### Key Features

- 🔬 **Multi-class Drug-Gene Interaction Prediction** (4 classes: No Link, Binding, Expression Increase/Decrease)
- 🧠 **Advanced GNN Architecture** using RGCN and RGAT models with ensemble learning
- 🤖 **AI-Powered Explanations** via Google Gemini integration
- 📊 **Interactive Web Interface** with bilingual support (Arabic/English)
- 📈 **High Performance**: 89.7% accuracy, 94.2% macro-AUC
- 📄 **PDF Report Generation** for analysis results

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AlzKB Graph   │───▶│   GNN Models     │───▶│  LLM Interface  │
│   (234K nodes)  │    │  (RGCN + RGAT)   │    │   (Gemini AI)   │
│  (1.6M edges)   │    │   Ensemble       │    │   Explanations  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Memgraph DB    │    │  PyTorch Geo     │    │  Streamlit UI   │
│   Docker        │    │  Deep Learning   │    │  Web Interface  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (for Memgraph database)
- CUDA-compatible GPU (optional, for faster training)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Mariam6600/Drug-repurposing-for-Alzheimer-s-disease.git
cd Drug-repurposing-for-Alzheimer-s-disease
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file (see .env.example)
GEMINI_API_KEY=your_gemini_api_key_here
MEMGRAPH_HOST=127.0.0.1
MEMGRAPH_PORT=7687
```

5. **Start Memgraph database**
```bash
# Pull and run Memgraph Docker container
docker run -it -p 7687:7687 -p 7444:7444 --name memgraph memgraph/memgraph

# Load AlzKB dataset (in separate terminal)
docker cp alzkb-v2-0-0_memgraph.cypherl memgraph:/tmp/
docker exec -i memgraph bash -c "mgconsole < /tmp/alzkb-v2-0-0_memgraph.cypherl"
```

6. **Run the application**
```bash
streamlit run Interface.py
```

## 📊 Dataset

The project uses the **AlzKB (Alzheimer's Knowledge Base)** containing:

- **234,037 nodes**: Genes, Drugs, Diseases, Pathways, etc.
- **1,668,487 relationships**: Various biological interactions
- **Focus**: Alzheimer's disease-related entities and interactions

### Key Statistics
- **Genes**: 193,279 nodes
- **Drugs**: 16,581 nodes  
- **Diseases**: 34 Alzheimer-related conditions
- **Drug-Gene Relations**: 65,490 interactions (3 types)

## 🔬 Methodology

### 1. Knowledge Graph Processing
- Extract Alzheimer-related subgraph from AlzKB
- Generate node embeddings using DeepWalk
- Create feature matrices with degree centrality and PageRank

### 2. GNN Architecture
```python
# Advanced RGCN Model
class Advanced_RGCN(nn.Module):
    - RGCNConv layers with residual connections
    - Layer normalization and dropout
    - Multi-layer edge decoder MLP

# Advanced RGAT Model  
class Advanced_RGAT(nn.Module):
    - Multi-head attention mechanism
    - Relational graph attention
    - Ensemble with RGCN for final predictions
```

### 3. Multi-Class Prediction
- **Class 0**: NO_LINK
- **Class 1**: CHEMICALBINDSGENE  
- **Class 2**: CHEMICALINCREASESEXPRESSION
- **Class 3**: CHEMICALDECREASESEXPRESSION

### 4. LLM Integration
- Google Gemini API for scientific explanations
- **Constrained Knowledge Approach**: The LLM is explicitly constrained to rely only on the provided graph-derived context and metadata, reducing hallucination and external knowledge leakage
- **Metadata-Driven Analysis**: Explanations are generated based solely on:
  - Drug class information from AlzKB
  - Gene functional annotations (GO terms, pathways)
  - Computational similarity patterns in representation space
- Bilingual support (Arabic/English)
- Fallback to local explanations if API fails

## 📈 Performance

| Model | Accuracy | Macro-AUC | F1-Score |
|-------|----------|-----------|----------|
| RGCN  | 87.3%    | 92.1%     | 0.851    |
| RGAT  | 88.9%    | 93.7%     | 0.867    |
| **Ensemble** | **89.7%** | **94.2%** | **0.879** |

## 🖥️ Web Interface

The Streamlit application provides:

- **Drug Selection**: Choose from 16,000+ drugs
- **Prediction Engine**: Real-time GNN inference
- **AI Explanations**: Contextual scientific interpretations
- **Report Generation**: PDF export functionality
- **Bilingual Support**: Arabic and English interfaces

### Interface Features
- Interactive drug-gene relationship exploration
- Confidence scores and probability distributions
- Scientific explanation generation
- Report basket for multiple analyses
- PDF export with professional formatting

## 📁 Project Structure

```
Drug-repurposing-for-Alzheimer-s-disease/
├── Alzheimer's disease _exp10_ALZKB.ipynb  # Main analysis notebook
├── Interface.py                            # Streamlit web application
├── requirements.txt                        # Python dependencies
├── .gitignore                             # Git ignore file
├── data/                                  # Generated data files
│   ├── alz_drugs_list.csv                # Extracted drug entities
│   ├── alz_genes_list.csv                # Extracted gene entities
│   ├── alz_graph_data.pt                 # Processed graph data
│   ├── rgcn_multi.pt                     # Trained RGCN model weights
│   ├── rgat_multi.pt                     # Trained RGAT model weights
│   ├── drug_metadata.json               # Drug class information
│   └── gene_metadata.json               # Gene function annotations
├── results/                              # Model outputs and visualizations
│   ├── RGCN_confusion_matrix.png        # RGCN performance metrics
│   ├── RGAT_confusion_matrix.png        # RGAT performance metrics
│   ├── Ensemble_confusion_matrix.png    # Ensemble results
│   └── *_learning_curves.png           # Training progress plots
└── README.md                            # This file
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Required: Gemini API Key for LLM explanations
GEMINI_API_KEY=your_api_key_here

# Database Configuration
MEMGRAPH_HOST=127.0.0.1
MEMGRAPH_PORT=7687

# Optional: Custom data paths
DATA_PATH=/path/to/your/data
```

### Model Configuration
- **Hidden Dimensions**: 256
- **Attention Heads**: 4 (RGAT)
- **Dropout Rate**: 0.3
- **Learning Rate**: 0.001
- **Ensemble Weights**: 70% RGCN + 30% RGAT

## 🧪 Usage Examples

### 1. Basic Prediction
```python
# Select a drug and predict interactions
drug_name = "Donepezil"
predictions = predict_interaction_with_embeddings(drug_id)

# Results include confidence scores for all 4 classes
for pred in predictions:
    print(f"Gene: {pred['gene']}, Relation: {pred['class_name']}, 
          Confidence: {pred['prob']:.3f}")
```

### 2. AI Explanation Generation
```python
# Generate scientific explanation
explanation = try_all_gemini_models(
    drug_name="Donepezil",
    gene_name="APOE", 
    class_id=2,  # CHEMICALINCREASESEXPRESSION
    class_prob=0.87,
    lang="English"
)
```

### 3. Batch Analysis
```python
# Analyze multiple drug-gene pairs
results = []
for drug in selected_drugs:
    predictions = predict_interaction_with_embeddings(drug)
    results.extend(predictions)

# Export to PDF report
generate_pdf_report(results, language="English")
```

## 🔬 Research Applications

This system can be used for:

- **Drug Repurposing & Hypothesis Generation**: Identify existing drugs with potential relevance to Alzheimer's disease and generate testable biological hypotheses
- **Target Discovery**: Find novel gene targets for drug development  
- **Mechanism Understanding**: Explain drug-gene interaction mechanisms
- **Literature Review**: Accelerate systematic review processes

## ⚠️ Important Notes

### Limitations
- **Computational Predictions**: All results require experimental validation
- **Knowledge Graph Completeness**: Limited by available data in AlzKB
- **Model Scope**: Trained specifically on Alzheimer's disease-related subgraph data
- **API Dependencies**: LLM explanations require internet connectivity
- **Hypothesis Generation Only**: Results represent computational hypotheses, not validated therapeutic recommendations

### Ethical Considerations
- Results are for research purposes only
- Not intended for clinical decision-making
- Requires expert interpretation and validation
- Potential biases from training data should be considered
- All predictions represent computational hypotheses requiring rigorous experimental validation

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black . && isort .
```

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{alzheimer_gnn_llm_2026,
  title={Knowledge Graph-Enhanced Drug-Gene Interaction Prediction in Alzheimer's Disease: A GNN-LLM Approach},
  author={Mariam Abdul aal},
  year={2025-2026},
  howpublished={GitHub Repository},
  url={https://github.com/Mariam6600/Drug-repurposing-for-Alzheimer-s-disease},
  note={Computational framework for drug-gene interaction analysis and hypothesis generation}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **AlzKB Team** for providing the Alzheimer's knowledge base
- **PyTorch Geometric** for graph neural network implementations  
- **Memgraph** for high-performance graph database
- **Google** for Gemini API access
- **Streamlit** for the web application framework

## 📞 Contact

- **Author**: Mariam Abdul aal
- **Email**: abdmariam900@gmail.com
- **Project Link**: [https://github.com/Mariam6600/Drug-repurposing-for-Alzheimer-s-disease](https://github.com/Mariam6600/Drug-repurposing-for-Alzheimer-s-disease)

---

**⚠️ Research Disclaimer**: This software is a computational research tool designed for hypothesis generation and academic investigation only. All predictions represent computational hypotheses derived from knowledge graph analysis and require rigorous experimental validation before any clinical consideration. The system is not intended for clinical decision-making, therapeutic recommendations, or patient care. The authors assume no responsibility for any misuse of this software or misinterpretation of its computational predictions.
