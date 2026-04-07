#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:09:08 2026

@author: baratachinuq
"""

import streamlit as st
import tifffile as tiff
import numpy as np
import tempfile
import os

st.set_page_config(page_title="UHV TIFF Stack Viewer", layout="wide")

st.title("🔬 Scientific TIFF Stack Viewer")
st.write("Upload a 32-bit TIFF stack to crop (1024x1024) and view with smart contrast.")

# 1. File Uploader
uploaded_file = st.file_uploader("Choose a TIFF file", type=['tif', 'tiff'])

if uploaded_file is not None:
    # Save uploaded bytes to a temporary file to enable Memory Mapping
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        # 2. Open via Memory Map (Very RAM efficient)
        with tiff.TiffFile(tmp_path) as tif:
            mmap_stack = tif.asarray(out='memmap')
            n_frames, h, w = mmap_stack.shape
            
            st.sidebar.info(f"Original Dimensions: {h}x{w}")
            st.sidebar.info(f"Total Frames: {n_frames}")

            # 3. Setup Cropping
            target = 1024
            sh = (h - target) // 2 if h > target else 0
            sw = (w - target) // 2 if w > target else 0

            # 4. Interactive Slider
            idx = st.slider("Select Frame", 0, n_frames - 1, 0)

            # 5. Process Frame (Crop + Smart Contrast)
            # Pull only the cropped chunk of the selected frame from disk
            frame = mmap_stack[idx, sh:sh+target, sw:sw+target].astype(np.float32)
            
            # Calculate 1% and 99% percentiles for this frame
            vmin = np.percentile(frame, 1)
            vmax = np.percentile(frame, 99)
            
            # Normalize to 0-1 range for browser display
            if vmax > vmin:
                display_frame = np.clip(frame, vmin, vmax)
                display_frame = (display_frame - vmin) / (vmax - vmin)
            else:
                display_frame = frame

            # 6. Display
            st.image(display_frame, caption=f"Frame {idx+1}/{n_frames} | Contrast Limits: {vmin:.2f} - {vmax:.2f}", use_container_width=True)

    finally:
        # Cleanup temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

else:
    st.info("Waiting for a file upload...")