import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from segment_anything import sam_model_registry, SamPredictor,SamAutomaticMaskGenerator
import cv2
import supervision as sv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw
import torch
from st_pages import Page, show_pages, add_page_title

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    # Remove color from rest of the image
    rest_image = np.ones_like(mask_image) * (1 - mask.reshape(h, w, 1))  # Inverse mask
    masked_image = mask_image + rest_image
    ax.imshow(masked_image)
    
def get_ellipse_coords(point: tuple[int, int]) -> tuple[int, int, int, int]:
    center = point
    radius = 2
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )

sam_checkpoint = "./sam_vit_b_01ec64.pth"
model_type = "vit_b"
torch.cuda.empty_cache()
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

st.header("Segmentation avec des point")

if 'points' not in st.session_state:
    st.session_state['points'] = []


image = st.file_uploader("Upload votre image :",type=['png', 'jpg'])
if image is not None:
    # Convert the uploaded file to an OpenCV-compatible format
    file_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Process the image using OpenCV
    image_ = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(image_)
    image_copy = im_pil.copy()
    draw = ImageDraw.Draw(image_copy)
    
    # Display the processed image
    if im_pil is not None:    
        value = streamlit_image_coordinates(im_pil,key="pil")

        if value is not None:
            point = value["x"], value["y"]

            if point not in st.session_state["points"]:
                st.session_state["points"].append(point)
                

    
    
    for point in st.session_state["points"]:
        coords = get_ellipse_coords(point)
        draw.ellipse(coords, fill="red")
        
            
    if st.button("Voir points"):
        st.image(image_copy)

    if st.button("Recommencer"):
        del st.session_state["points"]
        image_copy = im_pil.copy()
        draw = ImageDraw.Draw(image_copy)
        st.session_state["points"] =[]

    if st.button("Lancer segmentation") and image_ is not None:
        predictor.set_image(image_)


        masks, scores, _ = predictor.predict(
        point_coords=np.array(st.session_state["points"]),
        point_labels=np.array(np.ones(len(st.session_state["points"]))),
        multimask_output=True)

        sort_masks = list(zip(masks, scores))
        sort_masks.sort(key=lambda x:x[1],reverse=True)
        unzip = [[i for i, j in sort_masks],[j for i, j in sort_masks]]
        masks,scores = unzip[0],unzip[1]
        
        for i, (mask, score) in enumerate(zip(masks, scores)):
            width, height = im_pil.size
            px = 1/plt.rcParams['figure.dpi'] 
            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(width*px, height*px))
            ax.imshow(image_)
            show_mask(mask, ax)
            ax.axis('off')
            st.write(f"Mask {i+1}, Score: {score:.3f}")
            # Show the figure using Streamlit
            st.pyplot(fig)
        
    
            
    