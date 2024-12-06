import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from embedding import img2vec, text2vec

# Custom vector similarity function (cosine similarity)
def cosine_similarity(vec1, vec2):
    vec1 = np.squeeze(vec1)  # Ensure 1D
    vec2 = np.squeeze(vec2)  # Ensure 1D
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Connect to the SQLite database
conn = sqlite3.connect("products.db")
c = conn.cursor()

# Display all products with delete functionality
def display_products():
    st.title("Product Viewer")

    # Fetch all products
    c.execute("SELECT * FROM products")
    products = c.fetchall()

    if products:
        df = pd.DataFrame(
            products,
            columns=[
                "Product ID",
                "Name",
                "Description",
                "Stock Count",
                "Price",
                "Image",
                "Image Embedding",
                "Name Description Embedding",
            ],
        )

        # Display products in a table
        for _, row in df.iterrows():
            st.subheader(row["Name"])
            st.write(f"**Product ID**: {row['Product ID']}")
            st.write(f"**Description**: {row['Description']}")
            st.write(f"**Stock Count**: {row['Stock Count']}")
            st.write(f"**Price**: ${row['Price']:.2f}")
            
            # Decode and display the image
            image_data = base64.b64decode(row["Image"])
            image = Image.open(BytesIO(image_data))
            st.image(image, caption=row["Name"], use_container_width=True)
            
            # Add a delete button for each product
            if st.button(f"Delete {row['Name']}", key=row["Product ID"]):
                delete_product(row["Product ID"])
                st.success(f"Product '{row['Name']}' has been deleted.")
                st.rerun()  # Refresh the app to reflect changes
            st.write("---")
    else:
        st.warning("No products available in the database.")

# Function to delete a product from the database
def delete_product(product_id):
    c.execute("DELETE FROM products WHERE product_id = ?", (product_id,))
    conn.commit()

# Image search feature
def search_by_image():
    st.title("Search by Image")

    uploaded_image = st.file_uploader("Upload an Image to Search", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        
        # Generate the query vector
        image_base64 = base64.b64encode(uploaded_image.read()).decode("utf-8")
        query_vector = img2vec(image_base64)

        # Search for similar products
        c.execute("SELECT * FROM products")
        products = c.fetchall()
        results = []

        for row in products:
            product_id, name, description, stock_count, price, image, img_emb, name_desc_emb = row
            stored_vector = np.frombuffer(img_emb, dtype=np.float32)
            # st.write(f"Query vector shape: {np.array(query_vector).shape}")
            # st.write(f"Stored vector shape: {np.array(stored_vector).shape}")

            similarity = cosine_similarity(query_vector, stored_vector)
            results.append((name, similarity))

        # Sort results by similarity and display
        results = sorted(results, key=lambda x: x[1], reverse=True)
        st.subheader("Search Results")
        for name, similarity in results:
            st.write(f"**{name}** - Similarity: {similarity:.2f}")

# Text search feature
def search_by_text():
    st.title("Search by Text")

    query = st.text_input("Enter a text query to search (Name/Description):")
    if query:
        query_vector = text2vec(query)  # This should now be (384,)

        # Fetch all products
        c.execute("SELECT * FROM products")
        products = c.fetchall()
        results = []

        for row in products:
            product_id, name, description, stock_count, price, image, img_emb, name_desc_emb = row

            # Convert stored vector from binary to NumPy array
            stored_vector = np.frombuffer(name_desc_emb, dtype=np.float32)

            # Compute similarity
            similarity = cosine_similarity(query_vector, stored_vector)
            results.append((name, similarity))

        # Sort and display results
        results = sorted(results, key=lambda x: x[1], reverse=True)
        st.subheader("Search Results")
        for name, similarity in results:
            st.write(f"**{name}** - Similarity: {similarity:.2f}")


# Main navigation
st.sidebar.title("Navigation")
options = ["View Products", "Search by Image", "Search by Text"]
choice = st.sidebar.radio("Go to", options)

if choice == "View Products":
    display_products()
elif choice == "Search by Image":
    search_by_image()
elif choice == "Search by Text":
    search_by_text()

# Close the connection
conn.close()
