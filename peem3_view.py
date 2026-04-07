#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:09:08 2026

@author: baratachinuq
"""
import streamlit as st
import numpy as np
import tifffile as tiff
import tempfile
import os
import cv2
import time
from skimage.registration import phase_cross_correlation
from skimage import exposure, filters
from scipy.ndimage import shift, sobel, gaussian_filter
import matplotlib.pyplot as plt

# --- 1. ESSENTIAL CONFIG ---
st.set_page_config(page_title="PEEM Analysis Suite", layout="wide")

# This ensures the sidebar is always the first thing defined
st.sidebar.title("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload 32-bit TIFF", type=['tif', 'tiff'])

# --- 2. CORE FUNCTIONS ---
def apply_guyader_filter(image, sigma):
    img_f = image.astype(np.float32)
    if sigma > 0:
        img_f = gaussian_filter(img_f, sigma=sigma)
    dx = sobel(img_f, axis=1) 
    dy = sobel(img_f, axis=0) 
    mag = np.sqrt(dx**2 + dy**2)
    mag_mean = np.mean(mag)
    if mag_mean > 0:
        dx /= mag_mean
        dy /= mag_mean
    return dx + 1j * dy

# --- 3. MAIN APP LOGIC ---
if uploaded_file is not None:
    # Save to temp file to enable memory mapping (RAM efficient)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    # Open stack and define dimensions globally for this block
    mmap_stack = tiff.memmap(tmp_path)
    n_frames, h, w = mmap_stack.shape # Standardized variable name
    
    target = 1024
    sh = (h - target) // 2 if h > target else 0
    sw = (w - target) // 2 if w > target else 0

    # --- SIDEBAR CONTROLS ---
    st.sidebar.markdown("---")
    st.sidebar.title("📺 Playback Controls")
    play_mode = st.sidebar.toggle("▶ Play Animation")
    speed = st.sidebar.slider("Speed (s/frame)", 0.01, 1.0, 0.1)

    # --- TABS ---
    tab_view, tab_drift, tab_fft = st.tabs(["🖼 Viewer", "📉 Drift Correction", "🌀 FFT Masking"])

    with tab_view:
        if "frame_idx" not in st.session_state:
            st.session_state.frame_idx = 0

        # Create centered layout
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            image_placeholder = st.empty()
            if not play_mode:
                st.session_state.frame_idx = st.slider("Frame", 0, n_frames - 1, st.session_state.frame_idx)
            
            # Smart Contrast logic
            def get_display_frame(idx):
                f = mmap_stack[idx, sh:sh+target, sw:sw+target].astype(np.float32)
                vmin, vmax = np.percentile(f, [1, 99])
                return np.clip((f - vmin) / (vmax - vmin), 0, 1) if vmax > vmin else f

            if play_mode:
                while play_mode:
                    image_placeholder.image(get_display_frame(st.session_state.frame_idx), use_container_width=True)
                    st.session_state.frame_idx = (st.session_state.frame_idx + 1) % n_frames
                    time.sleep(speed)
            else:
                image_placeholder.image(get_display_frame(st.session_state.frame_idx), use_container_width=True)

    with tab_drift:
        st.header("Robust Drift Correction")
        sigma_val = st.number_input("Gaussian Blur", 0.0, 5.0, 1.5)
        roi_size = st.selectbox("ROI Size", [128, 256, 512], index=1)
        
        if st.button("🚀 Run Alignment"):
            with st.spinner("Processing..."):
                # Reference variables defined inside the button click
                y1, y2 = (h-roi_size)//2, (h+roi_size)//2
                x1, x2 = (w-roi_size)//2, (w+roi_size)//2
                window = filters.window('hann', (roi_size, roi_size))
                
                cum_y, cum_x = [0.0], [0.0]
                ref_roi = apply_guyader_filter(mmap_stack[0, y1:y2, x1:x2], sigma_val) * window
                
                # Corrected loop using standardized n_frames
                for i in range(1, n_frames):
                    mov_roi = apply_guyader_filter(mmap_stack[i, y1:y2, x1:x2], sigma_val) * window
                    shift_vec, _, _ = phase_cross_correlation(ref_roi, mov_roi, upsample_factor=100)
                    cum_y.append(cum_y[-1] + shift_vec[0])
                    cum_x.append(cum_x[-1] + shift_vec[1])
                    ref_roi = mov_roi

                # Apply shift to full stack
                corrected = np.zeros_like(mmap_stack, dtype=np.float32)
                for i in range(n_frames):
                    corrected[i] = shift(mmap_stack[i], shift=(cum_y[i], cum_x[i]), order=3)

                st.session_state.corrected_data = corrected
                st.line_chart({"X": cum_x, "Y": cum_y})
                st.success("Done!")

    with tab_fft:
        st.header("FFT Analysis")
        # Reuse n_frames here safely
        f_idx = st.slider("FFT Frame", 0, n_frames - 1, 0)
        # ... [FFT logic here] ...

    # Cleanup temp file
    os.remove(tmp_path)

else:
    st.info("Please upload a TIFF file in the sidebar to begin.")
