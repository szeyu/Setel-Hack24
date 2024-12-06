import numpy as np
import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect("products.db", check_same_thread=False)
c = conn.cursor()

# Custom vector similarity function (cosine similarity)
def cosine_similarity(vec1, vec2):
    vec1 = np.squeeze(vec1)  # Ensure 1D
    vec2 = np.squeeze(vec2)  # Ensure 1D
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def _items_similarity(query_vector, type):
    # Search for similar products
    c.execute("SELECT * FROM products")
    products = c.fetchall()
    results = []

    for row in products:
        product_id, name, description, stock_count, price, image, img_emb, name_desc_emb = row
        if type == "image":
            stored_vector = np.frombuffer(img_emb, dtype=np.float32)
        else:
            stored_vector = np.frombuffer(name_desc_emb, dtype=np.float32)

        similarity = cosine_similarity(query_vector, stored_vector)
        results.append((row, similarity))

    # Sort results by similarity and display
    results = sorted(results, key=lambda x: x[1], reverse=True)

    return results

def vector_search(vector, type):
    results = _items_similarity(vector, type)
    return results