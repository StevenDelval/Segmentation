import streamlit as st
from segment_anything import sam_model_registry, SamPredictor,SamAutomaticMaskGenerator
import cv2
import supervision as sv
import matplotlib.pyplot as plt
import numpy as np
import torch
from st_pages import Page, show_pages, add_page_title

sam_checkpoint = "./sam_vit_b_01ec64.pth"
model_type = "vit_b"
torch.cuda.empty_cache()
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

st.header("Segmentation sur toute l'image")

image = st.file_uploader("Upload votre image :",type=['png', 'jpg'])
if image is not None:
    # Convert the uploaded file to an OpenCV-compatible format
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Process the image using OpenCV
    # (Example: convert the image to grayscale)
    image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the processed image
    st.image(image_)
    masks = mask_generator.generate(image)
    fig, ax = plt.subplots()
        
    # Display the image
    ax.imshow(image_)
    ax.axis('off')
    show_anns(masks)
    # Show the figure using Streamlit
    st.pyplot(fig)

