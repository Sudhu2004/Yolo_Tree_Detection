import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from io import BytesIO
import requests
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO
import streamlit as st
import requests
from PIL import Image
from io import BytesIO

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


# Define function to draw polygons (from the first code)
def draw_polygons_on_image(image_path, predictions):
    if image_path.startswith('http'):
        response = requests.get(image_path)
        image = Image.open(BytesIO(response.content))
    else:
        image = Image.open(image_path)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)

    class_info = {
        0: {'color': 'red', 'label': 'tree_real_area'},
        1: {'color': 'green', 'label': 'Sticker'}
    }

    for prediction in predictions:
        class_id = prediction['class_id']
        points = prediction['points']
        polygon = [(point['x'], point['y']) for point in points]
        poly_patch = patches.Polygon(polygon, linewidth=2, edgecolor=class_info[class_id]['color'],
                                     facecolor='none', label=class_info[class_id]['label'])
        ax.add_patch(poly_patch)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    ax.axis('off')
    st.pyplot(fig)

# Function to calculate polygon area
def calculate_polygon_area(points):
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i]['x'] * points[j]['y']
        area -= points[j]['x'] * points[i]['y']
    return abs(area) / 2.0

# Function to calculate real-world areas
def calculate_real_world_areas(json_data):
    tree_image_area = 0
    sticker_image_area = 0
    
    altitude = 200
    pixel_size = 1.33
    focal_length = 3.6
    pixel_size_in_meters = pixel_size * 1e-6
    area_per_pixel = (altitude * pixel_size_in_meters) / focal_length
    square_meter_to_square_feet = 10.7639
    scale_factor = (altitude * pixel_size_in_meters) / focal_length
    
    for entry in json_data:
        predictions = entry['predictions']['predictions']
        for prediction in predictions:
            points = prediction['points']
            area_in_pixels = calculate_polygon_area(points)
            real_world_area_meters = area_in_pixels * area_per_pixel
            real_world_area_feet = real_world_area_meters * square_meter_to_square_feet

            st.write(f"Detected segment class {prediction['class_id']} - Area: {real_world_area_feet:.2f} square feet")

            if prediction['class_id'] == 0:
                tree_image_area += area_in_pixels
            elif prediction['class_id'] == 1:
                sticker_image_area += area_in_pixels

    total_tree_real_area_meters = tree_image_area * area_per_pixel
    total_tree_real_area_feet = total_tree_real_area_meters * square_meter_to_square_feet
    
    st.write(f"Total Relative Tree Area: {total_tree_real_area_feet:.2f} square feet")

st.title("Calculating Trees Area")

page = st.sidebar.selectbox("Choose a page", ["Segment Detection", "Tree Detection"])


if page == "Segment Detection":
    st.header("Segment Detection with Polygon Overlay")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        image_path = f"temp_{uploaded_file.name}"
        image.save(image_path)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key = st.secrets["roboflowApi"]
        )

        result = client.run_workflow(
            workspace_name="m-s-ramaiah-institute-of-technology",
            workflow_id="custom-workflow-tod",
            images={"image": image_path}
        )

        predictions = result[0]['predictions']['predictions']
        draw_polygons_on_image(image_path, predictions)
        calculate_real_world_areas(result)

        os.remove(image_path)

# Tree detection page
elif page == "Tree Detection":
    st.header("Tree Detection using YOLO")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        image_path = f"temp_{uploaded_file.name}"
        image.save(image_path)

        st.image(image, caption="Uploaded Image", use_column_width=True)

        HOME = os.getcwd()  
        model_trained = YOLO(f'{HOME}/runs/detect/train/weights/best.pt')

        st.write("Running YOLO detection...")
        results = model_trained.predict(source=image_path, conf=0.10, save=True)[0]

        boxes = results.boxes.xyxy.cpu()

        iou_threshold = 0.5
        merged_boxes = merge_boxes(boxes, iou_threshold)

        st.write(f"A total of {len(merged_boxes)} trees have been detected")
        st.write("Visualizing bounding boxes.")
        fig_merged = draw_boxes(image, merged_boxes, color='g', label='Tree')
        st.pyplot(fig_merged)

        os.remove(image_path)

