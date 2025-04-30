# Wine Recommendation RAG System

A Retrieval-Augmented Generation (RAG) system that provides intelligent wine recommendations and information using vector similarity search and language model generation.

## Overview

This project implements a RAG system that combines:
- Vector-based similarity search using Qdrant
- Semantic text embeddings using Sentence Transformers
- Natural language generation using FLAN-T5
- A curated dataset of high-rated wines with detailed tasting notes

## Features

- Semantic search across wine descriptions and tasting notes
- Vector similarity-based wine recommendations
- Natural language generation for wine-related queries
- Support for complex wine-related questions and recommendations

## Technologies Used

- **Python 3.x**
- **Key Libraries:**
  - `pandas` - For data manipulation and management
  - `qdrant-client` - Vector similarity database
  - `sentence-transformers` - For generating text embeddings
  - `transformers` - For natural language generation
  - `torch` - Deep learning framework

## Dataset

The system uses a curated dataset of top-rated wines (`top_rated_wines.csv`) containing:
- Wine names
- Regions
- Varieties
- Ratings
- Detailed tasting notes

## Setup and Installation

1. Clone the repository
2. Install required dependencies:
```bash
pip install pandas qdrant-client sentence-transformers transformers torch
```

## Usage

1. Load and preprocess the wine dataset:
```python
import pandas as pd
df = pd.read_csv('top_rated_wines.csv')
df = df[df['variety'].notna()]
```

2. Initialize the vector database and encoder:
```python
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer('all-MiniLM-L6-v2')
qdrant = QdrantClient(":memory:")
```

3. Query the system for wine recommendations:
```python
# Example query
results = retrieve_context("Top wines from Australia")
```

## System Architecture

1. **Data Ingestion Layer**
   - Loads and processes wine dataset
   - Handles data cleaning and preparation

2. **Vector Search Layer**
   - Generates embeddings using Sentence Transformers
   - Stores and queries vectors using Qdrant

3. **Generation Layer**
   - Uses FLAN-T5 for natural language generation
   - Produces human-readable responses based on retrieved context
