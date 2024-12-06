import streamlit as st
import sqlite3
from embedding import img2vec, text2vec
import base64
import pandas as pd
import numpy as np
from io import BytesIO
from PIL import Image

# Connect to the database
conn = sqlite3.connect('products.db')
c = conn.cursor()

# Create table if not exists
c.execute('''CREATE TABLE IF NOT EXISTS products
             (product_id TEXT, name TEXT, description TEXT, stock_count INTEGER, price REAL, image BLOB, image_vector_embedding BLOB, name_description_vector_embedding BLOB)''')

# Fetch all products from the database
c.execute("SELECT * FROM products")
products = c.fetchall()

# Streamlit interface for adding products
st.title("Add Product")

product_id = st.text_input("Product ID")
name = st.text_input("Name")
description = st.text_area("Description")
stock_count = st.number_input("Stock Count", min_value=0, step=1)
price = st.number_input("Price", min_value=0.0, step=0.01)
image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if st.button("Add Product"):
    if product_id and name and description and stock_count and price and image:
        # Convert image to base64
        image_base64 = base64.b64encode(image.read()).decode('utf-8')
        
        # Get image vector embedding
        image_vector_embedding = img2vec(image_base64)
        
        # Get text vector embedding
        text_vector_embedding = text2vec(description)
        
        # Get name and description vector embedding
        name_description_vector_embedding = text2vec(name + " " + description)
        
        # Convert embeddings to bytes
        image_vector_embedding_bytes = np.array(image_vector_embedding, dtype=np.float32).tobytes()
        name_description_vector_embedding_bytes = np.array(name_description_vector_embedding, dtype=np.float32).tobytes()

        # Insert into database
        c.execute("INSERT INTO products (product_id, name, description, stock_count, price, image, image_vector_embedding, name_description_vector_embedding) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (product_id, name, description, stock_count, price, image_base64, image_vector_embedding_bytes, name_description_vector_embedding_bytes))
        conn.commit()
        
        st.success("Product added successfully!")
    else:
        st.error("Please fill all the fields")

# Close the connection
conn.close()
