import streamlit as st
from segment_anything import sam_model_registry, SamPredictor,SamAutomaticMaskGenerator
import cv2
import supervision as sv
import matplotlib.pyplot as plt
import numpy as np
import torch
from st_pages import Page, show_pages, add_page_title



# Specify what pages should be shown in the sidebar, and what their titles 
# and icons should be
show_pages(
    [
        Page("app.py", "Home"),
        Page("pages/Segmentation_sur_toute_l_image.py", "Segmentation sur toute l'image"),
        Page("pages/Segmentation_avec_des_point.py", "Segmentation avec des point"),
        
        
    ]
)
st.header("Hello")