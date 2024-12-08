# Vector Search from Scratch

This repository implements a vector search solution based on image and text embeddings. Users can search for similar products using an image or a textual description.

## Project Structure

```plaintext
.
├── assets/                 # Folder containing sample product images for demonstration
├── .gitignore              # Git ignore file
├── LICENSE                 # License information
├── README.md               # Project documentation
├── add_product.py          # Streamlit app for adding new products to the database
├── dbms_product.py         # Streamlit app for Database management system (DBMS) logic for product handling
├── embedding.py            # Functions to generate embeddings for images and text
├── product_recognision.py  # Functions for searching and recognizing products
├── products.db             # SQLite database for storing product information and embeddings
├── requirements.txt        # Python dependencies required for this project
└── vector_search.py        # Functions for vector-based search and similarity calculation
```

## Features
- Add Products: Add new products to the database, including name, description, price, and an image.
- Search by Image: Upload an image to find visually similar products using CLIP-based image embeddings.
- Search by Text: Enter text to search for semantically similar products using MiniLM-based text embeddings.
- Delete Products: Remove products from the database directly via the UI.

## Getting Started
Prerequisites
- Python 3.8 or higher
- Pip package manager

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/vector-search.git
cd vector-search
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the add product streamlit interface:
```bash
streamlit run add_product.py
```

4. Start the product search streamlit interface:
```bash
streamlit run dbms_product.py
```


## How It Works

- ### Embeddings:
  Images are converted to embeddings using the img2vec function based on CLIP.
  Text descriptions are converted to embeddings using the text2vec function based on MiniLM.

- ### Database:
  Product data (including embeddings) is stored in an SQLite database (products.db).

- ### Vector Search:
  The vector_search.py script uses cosine similarity to match query vectors against stored embeddings.

## Example Files
  The `assets/` directory includes sample product images you can use to test the app.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
