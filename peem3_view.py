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
import time

st.set_page_config(page_title="UHV PEEM Viewer", layout="wide")

# --- SIDEBAR CONTROLS ---
st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("Upload 32-bit TIFF", type=['tif', 'tiff'])

# Animation Settings
st.sidebar.markdown("---")
play_mode = st.sidebar.toggle("▶ Play Animation")
speed = st.sidebar.slider("Speed (seconds per frame)", 0.01, 2.0, 0.1)
loop_playback = st.sidebar.checkbox("Loop", value=True)

# --- MAIN INTERFACE ---
st.title("🔬 Scientific Stack Browser")

if uploaded_file is not None:
    # Save to temp file for memmap efficiency
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        with tiff.TiffFile(tmp_path) as tif:
            mmap_stack = tif.asarray(out='memmap')
            n_frames, h, w = mmap_stack.shape
            
            # Setup Cropping (1024x1024 center)
            target = 1024
            sh = (h - target) // 2 if h > target else 0
            sw = (w - target) // 2 if w > target else 0

            # Manual Slider (hidden or inactive during play)
            if "frame_idx" not in st.session_state:
                st.session_state.frame_idx = 0

            if not play_mode:
                st.session_state.frame_idx = st.slider("Manual Frame Selection", 0, n_frames - 1, st.session_state.frame_idx)

            # --- CENTERED IMAGE LAYOUT ---
            # Using 3 columns [1, 2, 1] puts the image in the middle ~50% of the screen
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                image_placeholder = st.empty()
                info_placeholder = st.empty()

                # Animation Loop
                if play_mode:
                    while play_mode:
                        # Pull current frame
                        f_idx = st.session_state.frame_idx
                        frame = mmap_stack[f_idx, sh:sh+target, sw:sw+target].astype(np.float32)
                        
                        # Smart Contrast (1% - 99%)
                        vmin, vmax = np.percentile(frame, [1, 99])
                        if vmax > vmin:
                            display_frame = np.clip(frame, vmin, vmax)
                            display_frame = (display_frame - vmin) / (vmax - vmin)
                        else:
                            display_frame = frame

                        # Update UI
                        image_placeholder.image(display_frame, use_container_width=True)
                        info_placeholder.caption(f"Playing: Frame {f_idx + 1}/{n_frames} | {speed}s delay")
                        
                        # Increment frame
                        if st.session_state.frame_idx < n_frames - 1:
                            st.session_state.frame_idx += 1
                        elif loop_playback:
                            st.session_state.frame_idx = 0
                        else:
                            break # Stop if not looping
                        
                        time.sleep(speed)
                
                else:
                    # Static View (Manual slider mode)
                    f_idx = st.session_state.frame_idx
                    frame = mmap_stack[f_idx, sh:sh+target, sw:sw+target].astype(np.float32)
                    vmin, vmax = np.percentile(frame, [1, 99])
                    display_frame = np.clip(frame, vmin, vmax)
                    display_frame = (display_frame - vmin) / (vmax - vmin)
                    
                    image_placeholder.image(display_frame, use_container_width=True)
                    info_placeholder.caption(f"Static: Frame {f_idx + 1}/{n_frames} | Limits: {vmin:.1f} - {vmax:.1f}")

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
else:
    st.info("Please upload a file in the sidebar to begin.")