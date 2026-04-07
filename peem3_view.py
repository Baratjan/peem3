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

# --- 1. CONFIG & UI SETUP ---
st.set_page_config(page_title="PEEM Analysis Suite", layout="wide")
st.sidebar.title("📁 Data Input")
uploaded_file = st.sidebar.file_uploader("Upload 32-bit TIFF", type=['tif', 'tiff'])

# Constants from your local code
UPSAMPLE_FACTOR = 100

def apply_guyader_filter(image, sigma):
    """Replicates Align_ComplexEdgeFiltering.class."""
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

# --- 2. MAIN APP LOGIC ---
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    mmap_stack = tiff.memmap(tmp_path)
    n_frames, h, w = mmap_stack.shape
    
    # Global Crop (1024x1024)
    target = 1024
    sh = (h - target) // 2 if h > target else 0
    sw = (w - target) // 2 if w > target else 0

    st.sidebar.markdown("---")
    st.sidebar.title("📺 Playback Controls")
    play_mode = st.sidebar.toggle("▶ Play Animation")
    speed = st.sidebar.slider("Speed (s/frame)", 0.01, 1.0, 0.1)

    tab_view, tab_drift, tab_fft = st.tabs(["🖼 Viewer", "📉 Drift Correction", "🌀 FFT Masking"])

    # --- TAB 1: VIEWER ---
    with tab_view:
        if "frame_idx" not in st.session_state:
            st.session_state.frame_idx = 0
        
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            image_placeholder = st.empty()
            if not play_mode:
                st.session_state.frame_idx = st.slider("Frame", 0, n_frames - 1, st.session_state.frame_idx)
            
            def get_display_frame(idx, stack_data):
                f = stack_data[idx, sh:sh+target, sw:sw+target].astype(np.float32)
                vmin, vmax = np.percentile(f, [1, 99])
                return np.clip((f - vmin) / (vmax - vmin), 0, 1) if vmax > vmin else f

            if play_mode:
                while play_mode:
                    image_placeholder.image(get_display_frame(st.session_state.frame_idx, mmap_stack), use_container_width=True)
                    st.session_state.frame_idx = (st.session_state.frame_idx + 1) % n_frames
                    time.sleep(speed)
            else:
                image_placeholder.image(get_display_frame(st.session_state.frame_idx, mmap_stack), use_container_width=True)

    # --- TAB 2: DRIFT CORRECTION ---
    with tab_drift:
        st.header("Robust Drift Correction")
        col_ctrl, col_prev = st.columns([1, 2])
        
        with col_ctrl:
            sigma_val = st.sidebar.number_input("Filter: Gaussian Blur Radius", 0.0, 20.0, 1.5)
            st.markdown("### 🎯 Region of Interest (ROI)")
            roi_size = st.selectbox("ROI Square Size (Power of 2)", [128, 256, 512], index=1)
            roi_x = st.slider("ROI Center X", 0, w, w//2)
            roi_y = st.slider("ROI Center Y", 0, h, h//2)
            
            run_drift = st.button("🚀 Run Alignment")

        with col_prev:
            # Preview ROI Selection
            y1, y2 = max(0, roi_y - roi_size//2), min(h, roi_y + roi_size//2)
            x1, x2 = max(0, roi_x - roi_size//2), min(w, roi_x + roi_size//2)
            
            ref_frame = mmap_stack[0].copy()
            preview = exposure.rescale_intensity(ref_frame, out_range=(0, 255)).astype(np.uint8)
            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_GRAY2RGB)
            cv2.rectangle(preview_rgb, (x1, y1), (x2, y2), (255, 0, 0), 5)
            st.image(preview_rgb, caption="Alignment ROI (Red Box)", use_container_width=True)

        if run_drift:
            with st.spinner("Aligning stack..."):
                cum_y, cum_x = [0.0], [0.0]
                window = filters.window('hann', (y2-y1, x2-x1))
                ref_roi = apply_guyader_filter(mmap_stack[0, y1:y2, x1:x2], sigma_val) * window
                
                for i in range(1, n_frames):
                    mov_roi = apply_guyader_filter(mmap_stack[i, y1:y2, x1:x2], sigma_val) * window
                    shift_vec, _, _ = phase_cross_correlation(ref_roi, mov_roi, upsample_factor=UPSAMPLE_FACTOR)
                    cum_y.append(cum_y[-1] + shift_vec[0])
                    cum_x.append(cum_x[-1] + shift_vec[1])
                    ref_roi = mov_roi

                # Apply shift and save result to session state
                corrected = np.zeros_like(mmap_stack, dtype=np.float32)
                for i in range(n_frames):
                    corrected[i] = shift(mmap_stack[i], shift=(cum_y[i], cum_x[i]), order=3, mode='constant', cval=0)

                st.session_state.corrected_data = corrected
                st.session_state.cum_x, st.session_state.cum_y = cum_x, cum_y
                st.success("Alignment Complete!")

        if 'corrected_data' in st.session_state:
            st.markdown("---")
            st.subheader("Aligned Stack Results")
            st.line_chart({"X Drift": st.session_state.cum_x, "Y Drift": st.session_state.cum_y})
            
            # Export Aligned Stack
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as out_tmp:
                tiff.imwrite(out_tmp.name, st.session_state.corrected_data, photometric='minisblack')
                with open(out_tmp.name, "rb") as f:
                    st.download_button("💾 Download Aligned 32-bit Stack", f, "aligned_stack.tif")

    # --- TAB 3: FFT RECONSTRUCT ---
    with tab_fft:
        st.header("Interactive FFT Spot-Masking")
        f_idx = st.slider("Select Frame for Analysis", 0, n_frames - 1, 0, key="fft_frame")
        
        img_raw = mmap_stack[f_idx].astype(np.float64)
        dft_shift = np.fft.fftshift(np.fft.fft2(img_raw))
        mag_log = np.log1p(np.abs(dft_shift))
        
        col_fft, col_recon = st.columns(2)
        
        with col_fft:
            st.markdown("### FFT Domain")
            # Display FFT with percentile scaling
            lo, hi = np.percentile(mag_log, [2, 98])
            mag_disp = np.clip((mag_log - lo) / (hi - lo), 0, 1)
            st.image(mag_disp, caption="FFT (Log Scale)", use_container_width=True)
            
            mode = st.radio("Mode", ["INCLUDE (Bandpass)", "EXCLUDE (Notch)"], horizontal=True)
            radius = st.slider("Mask Radius", 2, 120, 15)
            coords = st.text_input("Enter Spot Coordinates (x,y ; x,y)", "512,512")

        with col_recon:
            st.markdown("### Spatial Reconstruction")
            mask = np.zeros_like(img_raw) if "INCLUDE" in mode else np.ones_like(img_raw)
            fill_val = 1.0 if "INCLUDE" in mode else 0.0
            
            # Add spot circles to mask
            try:
                for coord in coords.split(';'):
                    cx, cy = map(int, coord.strip().split(','))
                    cv2.circle(mask, (cx, cy), radius, fill_val, -1)
                    # Friedel conjugate
                    cv2.circle(mask, (w - cx, h - cy), radius, fill_val, -1)
            except:
                st.warning("Coordinate format: x,y ; x,y")

            recon = np.abs(np.fft.ifft2(np.fft.ifftshift(dft_shift * mask)))
            st.image(recon / (recon.max() + 1e-8), caption="Inverse FFT Reconstruction", use_container_width=True)
            
            if st.button("💾 Download Reconstruction"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as f_out:
                    tiff.imwrite(f_out.name, recon.astype(np.float32))
                    with open(f_out.name, "rb") as f:
                        st.download_button("Confirm Download", f, "fft_reconstructed.tif")

    os.remove(tmp_path)
else:
    st.info("Upload a file in the sidebar to begin.")
