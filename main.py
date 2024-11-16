import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import os

# Define functions for IoU and box merging
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    iou = intersection / union if union > 0 else 0

    return iou

def merge_boxes(boxes, iou_threshold=0.5):
    merged_boxes = []
    used = set()

    for i, box1 in enumerate(boxes):
        if i in used:
            continue
        merged = box1.clone()
        used.add(i)

        for j, box2 in enumerate(boxes):
            if j in used:
                continue
            iou = compute_iou(merged, box2)
            if iou > iou_threshold:
                merged[0] = min(merged[0], box2[0])
                merged[1] = min(merged[1], box2[1])
                merged[2] = max(merged[2], box2[2])
                merged[3] = max(merged[3], box2[3])
                used.add(j)

        merged_boxes.append(merged)

    return torch.stack(merged_boxes)

def draw_boxes(image, boxes, color='r', label=None):
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)

    for box in boxes:
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        if label:
            ax.text(x1, y1 - 5, label, color=color, fontsize=12, backgroundcolor="white")

    plt.axis('off')
    return fig

# Streamlit UI
st.title("Detecting Number of Trees")
st.write("Upload an image to detect objects and merge bounding boxes.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save and load the uploaded image
    image = Image.open(uploaded_file)
    image_path = f"temp_{uploaded_file.name}"
    image.save(image_path)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Load YOLO model
    st.write("Loading YOLO model...")
    HOME = os.getcwd() # Update if running locally
    model_trained = YOLO(f'{HOME}/runs/detect/train/weights/best.pt')

    # Run YOLO prediction
    st.write("Running YOLO detection...")
    results = model_trained.predict(source=image_path, conf=0.10, save=True)[0]

    # Extract bounding boxes
    boxes = results.boxes.xyxy.cpu()

    # Merge overlapping boxes
    st.write("Merging overlapping bounding boxes...")
    iou_threshold = 0.5
    merged_boxes = merge_boxes(boxes, iou_threshold)

    # # Visualize original boxes
    # st.write("Visualizing original bounding boxes...")
    # fig_original = draw_boxes(image, boxes, color='r', label='Original')
    # st.pyplot(fig_original)

    # Visualize merged boxes
    st.write(f"A total of {len(merged_boxes)} trees have been detected")
    st.write("Visualizing bounding boxes.")
    fig_merged = draw_boxes(image, merged_boxes, color='g', label='Tree')
    st.pyplot(fig_merged)

    # Clean up temporary file
    os.remove(image_path)
