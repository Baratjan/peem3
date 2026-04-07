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
from skimage.registration import phase_cross_correlation
from skimage import exposure, filters
from scipy.ndimage import shift, sobel, gaussian_filter
import matplotlib.pyplot as plt

# --- CONFIGURATION & CORE LOGIC FROM YOUR CODE ---
CROP_MARGIN = 10
UPSAMPLE_FACTOR = 100

def apply_guyader_filter(image, sigma):
    """Your specific complex edge filtering logic."""
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

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="PEEM Analysis Suite", layout="wide")
st.title("🔬 PEEM Robust Analysis Suite")

# 1. File Handling
uploaded_file = st.sidebar.file_uploader("Upload 32-bit TIFF", type=['tif', 'tiff'])

if uploaded_file:
    # Save to temp for memmap
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    mmap_stack = tiff.memmap(tmp_path)
    n_frames, h, w = mmap_stack.shape

    # 2. Tabs for Different Workflows
    tab_view, tab_drift, tab_fft = st.tabs(["🖼 Viewer", "📉 Drift Correction", "🌀 FFT Masking"])

    # --- TAB 1: VIEWER ---
    with tab_view:
        # [Insert your previously optimized playback/viewer code here]
        st.info("Use the sidebar to play or scroll through the raw stack.")

    # --- TAB 2: DRIFT CORRECTION ---
    with tab_drift:
    st.header("Robust Drift Correction (Guyader Method)")
    
    col_ctrl, col_prev = st.columns([1, 2])
    
    with col_ctrl:
        sigma_val = st.number_input("Gaussian Blur Radius", 0.0, 10.0, 1.5)
        st.markdown("### Select Tracking ROI")
        roi_size = st.selectbox("ROI Square Size (Power of 2)", [64, 128, 256, 512], index=1)
        roi_x = st.slider("ROI Center X", 0, w, w//2)
        roi_y = st.slider("ROI Center Y", 0, h, h//2)
        
        run_drift = st.button("🚀 Run Alignment", use_container_width=True)

    with col_prev:
        y1, y2 = max(0, roi_y - roi_size//2), min(h, roi_y + roi_size//2)
        x1, x2 = max(0, roi_x - roi_size//2), min(w, roi_x + roi_size//2)
        
        # Live Preview of Tracking Box
        preview = exposure.rescale_intensity(mmap_stack[0], out_range=(0, 255)).astype(np.uint8)
        preview = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB)
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 0, 0), 5)
        st.image(preview, caption="Tracking ROI Preview (Red Box)", use_container_width=True)

    if run_drift:
        with st.spinner("Aligning stack using Robust Sequential Logic..."):
            # 1. Setup based on your drift_correct.ipynb
            num_frames = n_frames
            cum_y, cum_x = [0.0], [0.0]
            roi_h, roi_w = y2-y1, x2-x1
            window = filters.window('hann', (roi_h, roi_w))
            
            # Pre-calculate complex edge filter for frames
            # Processing frame-by-frame to conserve RAM
            ref_roi = apply_guyader_filter(mmap_stack[0, y1:y2, x1:x2], sigma_val) * window
            
            for i in range(1, num_frames):
                mov_roi = apply_guyader_filter(mmap_stack[i, y1:y2, x1:x2], sigma_val) * window
                
                # Phase Cross Correlation
                shift_vec, error, diffphase = phase_cross_correlation(
                    ref_roi, mov_roi, upsample_factor=UPSAMPLE_FACTOR
                )
                
                cum_y.append(cum_y[-1] + shift_vec[0])
                cum_x.append(cum_x[-1] + shift_vec[1])
                ref_roi = mov_roi # Sequential logic: next frame compared to current

            # 2. Apply Bi-Cubic Correction to the full stack
            corrected = np.zeros_like(mmap_stack, dtype=np.float32)
            for i in range(num_frames):
                corrected[i] = shift(mmap_stack[i], shift=(cum_y[i], cum_x[i]), order=3, mode='constant', cval=0)

            # 3. Auto-Crop to overlapping region
            y_min, y_max = int(np.ceil(max(0, max(cum_y)))), int(np.floor(min(0, min(cum_y))))
            x_min, x_max = int(np.ceil(max(0, max(cum_x)))), int(np.floor(min(0, min(cum_x))))
            
            final_stack = corrected[:, 
                y_min : (corrected.shape[1] + y_max),
                x_min : (corrected.shape[2] + x_max)
            ]
            
            # Store results in Session State for persistence
            st.session_state.final_stack = final_stack
            st.session_state.cum_x = cum_x
            st.session_state.cum_y = cum_y
            
            st.success(f"Alignment Complete! New Shape: {final_stack.shape}")

    # --- THE DOWNLOAD SECTION ---
    if 'final_stack' in st.session_state:
        st.markdown("---")
        st.subheader("Export Results")
        
        # Display the drift profile chart
        st.line_chart({"X Shift": st.session_state.cum_x, "Y Shift": st.session_state.cum_y})
        
        # Create the download button
        with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp_out:
            tiff.imwrite(tmp_out.name, st.session_state.final_stack.astype(np.float32), photometric='minisblack')
            
            with open(tmp_out.name, "rb") as f:
                st.download_button(
                    label="💾 Download Aligned 32-bit Stack",
                    data=f,
                    file_name="aligned_peem_stack.tif",
                    mime="image/tiff",
                    use_container_width=True
                )
        
        # Cleanup temporary export file
        if os.path.exists(tmp_out.name):
            os.remove(tmp_out.name)
    
    # --- TAB 3: FFT MASKING ---
    with tab_fft:
        st.header("Interactive FFT Spot Masking")
        
        # 1. FFT Processing
        target_idx = st.slider("Select Frame for FFT", 0, n_frames-1, 0)
        img_raw = mmap_stack[target_idx].astype(np.float64)
        dft_shift = np.fft.fftshift(np.fft.fft2(img_raw))
        mag_log = np.log1p(np.abs(dft_shift))
        
        # Display FFT
        col_fft, col_recon = st.columns(2)
        
        with col_fft:
            st.markdown("### FFT Magnitude")
            # Percentile display logic from your live_fft_processor.py
            lo, hi = np.percentile(mag_log, [2, 98])
            mag_disp = np.clip((mag_log - lo) / (hi - lo), 0, 1)
            st.image(mag_disp, use_container_width=True, caption="FFT (Log Scale)")
            
            # Mask Inputs
            mask_mode = st.radio("Mode", ["INCLUDE", "EXCLUDE"], horizontal=True)
            radius = st.slider("Mask Radius", 1, 100, 15)
            
            # Since we can't click, we use text input for coordinates
            coords_str = st.text_input("Enter Spot Coordinates (x,y; x,y...)", "512,512")

        with col_recon:
            st.markdown("### Reconstruction")
            # Apply Masking logic from your code
            mask = np.ones_like(img_raw) if mask_mode == "EXCLUDE" else np.zeros_like(img_raw)
            val = 0.0 if mask_mode == "EXCLUDE" else 1.0
            
            try:
                for coord in coords_str.split(';'):
                    cx, cy = map(int, coord.split(','))
                    cv2.circle(mask, (cx, cy), radius, val, -1)
                    # Friedel conjugate
                    rows, cols = img_raw.shape
                    cv2.circle(mask, (cols-cx, rows-cy), radius, val, -1)
            except:
                st.warning("Enter valid coordinates like: 400,300; 600,700")

            f_filtered = dft_shift * mask
            recon = np.abs(np.fft.ifft2(np.fft.ifftshift(f_filtered)))
            
            st.image(recon / recon.max(), use_container_width=True, caption="Inverse FFT Result")

    # Cleanup
    os.remove(tmp_path)
