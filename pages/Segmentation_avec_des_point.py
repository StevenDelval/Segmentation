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
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

st.header("Segmentation avec des point")

st.file_uploader("",type=['png', 'jpg'])