#importing Desired Libraries
import gradio as gr
from transformers import AutoProcessor,BlipForConditionalGeneration
from PIL import Image
import numpy as np

#Initialization of Processor and Model
processor=AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
 

def caption_image(input_image: np.ndarray):
    # Convert numpy array to PIL Image and convert to RGB
    raw_image = Image.fromarray(input_image).convert('RGB')
    
    # Process the image
    image=processor(images=raw_image,text="This image is ",return_tensors="pt")

    # Generate a caption for the image
    outputs=model.generate(**image,max_length=1000)

    # Decode the generated tokens to text and store it into `caption`
    caption=processor.decode(outputs[0],skip_special_tokens=True)
    
    return caption

iface=gr.Interface(
    fn=caption_image,
    inputs=gr.Image(),
    outputs="text",
    title="Image Captioning",
    description="This is a simple web app for generating captions for images using a trained model."
    )
iface.launch()