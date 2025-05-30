import streamlit as st
import cv2
import numpy as np
import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the xu_ly_anh directory
xu_ly_anh_dir = os.path.abspath(os.path.join(current_dir, '..', 'xu_li_anh'))

# Add the xu_ly_anh directory to the sys.path
sys.path.append(xu_ly_anh_dir)

import Chapter03 as c3
import Chapter04 as c4
import Chapter05 as c5
import Chapter09 as c9

def main():

    L = 256

    #st.set_page_config(page_title="Machine Vision", layout="wide")

    #st.title("Machine Vision Application")

    # Function to load image
    def load_image(color=False):
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'tif', 'bmp', 'gif', 'png'])
        if uploaded_file is not None:
            if color:
                return cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)
            else:
                return cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        return None

    # Function to display images
    def display_images(imgin, imgout=None):
        if imgin is not None:
            st.image(imgin, channels="BGR" if imgin.ndim == 3 else "GRAY", caption="Input Image")
        if imgout is not None:
            st.image(imgout, channels="BGR" if imgout.ndim == 3 else "GRAY", caption="Output Image")

    # Sidebar for file operations
    st.sidebar.header("File Operations")
    file_operation = st.sidebar.radio("Choose an operation", ["Open", "Open Color", "Save"])

    # Handling file operations
    imgin = None
    imgout = None

    if file_operation == "Open":
        imgin = load_image(color=False)
    elif file_operation == "Open Color":
        imgin = load_image(color=True)
    elif file_operation == "Save":
        if 'imgout' in globals() and imgout is not None:
            st.sidebar.image(imgout, channels="BGR" if imgout.ndim == 3 else "GRAY", caption="Result Image")
            st.sidebar.download_button(label="Download Image", data=cv2.imencode('.png', imgout)[1].tobytes(), file_name="output.png")

    # Sidebar for chapter operations
    st.sidebar.header("Chapter Operations")

    chapters = {
        "Chapter3": [
            "Negative", "Logarit", "PiecewiseLinear", "Power", "Histogram",
            "HistEqual", "HistEqualColor", "LocalHist", "HistStat",
            "BoxFilter", "LowpassGauss", "Threshold", "MedianFilter",
            "Sharpen", "Gradient"
        ],
        "Chapter4": ["Spectrum", "FrequencyFilter", "DrawNotchRejectFilter", "RemoveMoire"],
        "Chapter5": ["CreateMotionNoise", "DenoiseMotion", "DenoisestMotion"],
        "Chapter9": ["ConnectedComponent", "CountRice"]
    }

    chapter = st.sidebar.selectbox("Select Chapter", list(chapters.keys()))
    operation = st.sidebar.selectbox("Select Operation", chapters[chapter])

    # Initial display
    if imgin is not None:
        display_images(imgin)

    if st.sidebar.button("Run Operation"):
        imgout = None
        if imgin is not None or operation == "DrawNotchRejectFilter":
            if chapter == "Chapter3":
                if operation == "Negative":
                    imgout = c3.Negative(imgin)
                elif operation == "Logarit":
                    imgout = c3.Logarit(imgin)
                elif operation == "PiecewiseLinear":
                    imgout = c3.PiecewiseLinear(imgin)
                elif operation == "Power":
                    imgout = c3.Power(imgin)           
                elif operation == "Histogram":
                    imgout = c3.Histogram(imgin)
                elif operation == "HistEqual":
                    imgout = cv2.equalizeHist(imgin)
                elif operation == "HistEqualColor":
                    imgout = c3.HistEqualColor(imgin)
                elif operation == "LocalHist":
                    imgout = c3.LocalHist(imgin)
                elif operation == "HistStat":
                    imgout = c3.HistStat(imgin)
                elif operation == "BoxFilter":
                    imgout = cv2.blur(imgin, (21, 21))
                elif operation == "LowpassGauss":
                    imgout = cv2.GaussianBlur(imgin, (43, 43), 7.0)
                elif operation == "Threshold":
                    imgout = c3.Threshold(imgin)
                elif operation == "MedianFilter":
                    imgout = cv2.medianBlur(imgin, 7)
                elif operation == "Sharpen":
                    imgout = c3.Sharpen(imgin)
                elif operation == "Gradient":
                    imgout = c3.Gradient(imgin)
            elif chapter == "Chapter4":
                if operation == "Spectrum":
                    imgout = c4.Spectrum(imgin)
                elif operation == "FrequencyFilter":
                    imgout = c4.FrequencyFilter(imgin)
                elif operation == "DrawNotchRejectFilter":
                    imgout = c4.DrawNotchRejectFilter()
                elif operation == "RemoveMoire":
                    imgout = c4.RemoveMoire(imgin)
            elif chapter == "Chapter5":
                if operation == "CreateMotionNoise":
                    imgout = c5.CreateMotionNoise(imgin)
                elif operation == "DenoiseMotion":
                    imgout = c5.DenoiseMotion(imgin)
                elif operation == "DenoisestMotion":
                    temp = cv2.medianBlur(imgin, 7)
                    imgout = c5.DenoiseMotion(temp)
            elif chapter == "Chapter9":
                if operation == "ConnectedComponent":
                    imgout = c9.ConnectedComponent(imgin)
                elif operation == "CountRice":
                    imgout = c9.CountRice(imgin)
            
            display_images(None, imgout)
        else:
            st.warning("Please upload an image file to proceed.")


if __name__ == '__main__':
    main()
