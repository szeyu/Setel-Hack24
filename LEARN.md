# Learn: Understanding Vector Search from Scratch

This file provides a comprehensive guide to help you understand how the components of this project work together to implement vector search based on embeddings.

---

## Table of Contents

1. [What is Vector Search?](#what-is-vector-search)
2. [How Embeddings Work](#how-embeddings-work)
3. [Database and Storage](#database-and-storage)
4. [Search Algorithm](#search-algorithm)
5. [Code Walkthrough](#code-walkthrough)
6. [Extending the Project](#extending-the-project)

---

## What is Vector Search?

Vector search is a method of finding items similar to a query based on numerical representations (embeddings). Instead of relying on exact keyword matches, vector search uses distances (e.g., cosine similarity) between embeddings to measure similarity.

### Use Cases
- **Image Search**: Find similar images to a given one.
- **Text Search**: Retrieve products based on textual descriptions.

---

## How Embeddings Work

Embeddings are dense numerical representations of data (e.g., text or images). In this project:

- **Image Embeddings**: Created using CLIP (Contrastive Language–Image Pretraining) to capture visual similarities.
- **Text Embeddings**: Generated using MiniLM, a lightweight transformer model for semantic understanding.

Embeddings enable meaningful comparisons between diverse data types (e.g., text vs. image).

---

## Database and Storage

Product information, including embeddings, is stored in an SQLite database (`products.db`). The schema includes:

- **Product Name**: Descriptive name of the product.
- **Description**: Textual information about the product.
- **Image Path**: Location of the product image.
- **Price**: Numeric value indicating the cost of the product.
- **Embeddings**: Vector representations for both text and image data.

SQLite provides a lightweight yet robust solution for storing structured data.

---

## Search Algorithm

The core search functionality relies on vector similarity:

1. **Query Vector**:
   - For image input, `img2vec` generates a query embedding.
   - For text input, `text2vec` generates a query embedding.

2. **Cosine Similarity**:
   - Measures the angular similarity between the query embedding and stored embeddings.

3. **Result Ranking**:
   - Products are ranked based on similarity scores, and the most similar products are displayed.

---

## Code Walkthrough

### Key Files
1. **`embedding.py`**:
   - Functions to convert images and text to embeddings.

2. **`dbms_product.py`**:
   - Manages CRUD operations for the product database using Streamlit.

3. **`vector_search.py`**:
   - Implements vector search logic using cosine similarity.

4. **`add_product.py`**:
   - User interface to add new products, including name, description, price, and image upload.

5. **`product_recognision.py`**:
   - Backend for searching products by uploading an image or entering a description.

---

## Extending the Project

### Ideas for Improvement
- **Advanced Models**: Integrate newer models for embeddings like OpenAI's latest APIs.
- **Real-time Updates**: Implement a web API to handle dynamic product additions.
- **Improved UI/UX**: Add enhanced visualization for search results and rankings.
- **Cloud Deployment**: Deploy the application using services like AWS, GCP, or Azure.

### Customization
- Replace the current models with domain-specific ones (e.g., a medical image embedding model for healthcare applications).
- Integrate with a NoSQL database for larger-scale deployments.

---

## Resources for Learning

1. [CLIP: Contrastive Language–Image Pretraining](https://github.com/openai/CLIP)
2. [MiniLM: Lightweight and Efficient Transformer](https://github.com/microsoft/unilm)
3. [Streamlit Documentation](https://docs.streamlit.io/)
4. [SQLite Tutorial](https://sqlite.org/docs.html)
5. [Cosine Similarity Explanation](https://en.wikipedia.org/wiki/Cosine_similarity)

---

Happy Learning! 🚀
