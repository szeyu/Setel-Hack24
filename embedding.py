from transformers import CLIPProcessor, CLIPModel, pipeline
from PIL import Image
import base64
import io

# Function to convert image (base64 string) to vector
def img2vec(image_base64):
    # Decode the base64 string
    image_data = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_data))

    # Load the CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Process the image and get the vector
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    return outputs.detach().numpy()

# Function to convert text to vector using MiniLM
def text2vec(text):
    # Load the MiniLM model and tokenizer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    text_pipeline = pipeline("feature-extraction", model=model_name)

    # Get the vector for the text
    text_vector = text_pipeline(text)
    return text_vector[0]