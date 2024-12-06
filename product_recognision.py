from embedding import img2vec, text2vec
from vector_search import vector_search

def identify_item_from_image(image_data: str) -> tuple:
    """
    Convert image to vector embedding, search the vector DB, and return item_id with similarity.
    """
    image_vector = img2vec(image_data)  # Convert image to embedding
    results = vector_search(image_vector, type="image")  # Search vector DB
    top_result = results[0]  # Get the most similar item
    item_id, similarity = top_result[0][0], top_result[1]
    
    return item_id, similarity

def identify_item_from_text(text_data: str) -> tuple:
    """
    Convert text to vector embedding, search the vector DB, and return item_id with similarity.
    """
    text_vector = text2vec(text_data)  # Convert text to embedding
    results = vector_search(text_vector, type="text")  # Search vector DB
    top_result = results[0]  # Get the most similar item
    item_id, similarity = top_result[0][0], top_result[1]
    
    return item_id, similarity
