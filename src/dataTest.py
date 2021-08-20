import torch
from labelbox import Client
from labelbox.schema.project import Project
import requests
from getpass import getpass
from PIL import Image
import numpy as np
from io import BytesIO
from typing import Dict, Any
import numpy as np
import os
import cv2


def visualize_poly(image: np.ndarray, tool: Dict[str, Any]) -> np.ndarray:
    """
    Draws a polygon on an image
    
    Args:
        image (np.ndarray): image to draw a polygon onto
        tool (Dict[str,any]): Dict response from the export
    Returns:
        image with a polygon drawn on it.
    """
    poly = [[pt["x"], pt["y"]] for pt in tool["polygon"]]
    poly = np.array(poly)
    return cv2.polylines(image, [poly], True, (0, 255, 0), thickness=5)


def visualize_bbox(image: np.ndarray, tool: Dict[str, Any]) -> np.ndarray:
    """
    Draws a bounding box on an image
    
    Args:
        image (np.ndarray): image to draw a bounding box onto
        tool (Dict[str,any]): Dict response from the export
    Returns:
        image with a bounding box drawn on it.
    """
    start = (tool["bbox"]["left"], tool["bbox"]["top"])
    end = (tool["bbox"]["left"] + tool["bbox"]["width"],
           tool["bbox"]["top"] + tool["bbox"]["height"])
    return cv2.rectangle(image, start, end, (255, 0, 0), 5)


def visualize_point(image: np.ndarray, tool: Dict[str, Any]) -> np.ndarray:
    """
    Draws a point on an image
    
    Args:
        image (np.ndarray): image to draw a point onto
        tool (Dict[str,any]): Dict response from the export
    Returns:
        image with a point drawn on it.
    """
    return cv2.circle(image, (tool["point"]["x"], tool["point"]["y"]),
                      radius=10,
                      color=(0, 0, 255),
                      thickness=-1)


def visualize_mask(image: np.ndarray,
                   tool: Dict[str, Any],
                   alpha: float = 0.5) -> np.ndarray:
    """
    Overlays a mask onto an image
    
    Args:
        image (np.ndarray): image to overlay mask onto
        tool (Dict[str,any]): Dict response from the export
        alpha: How much to weight the mask when adding to the image
    Returns:
        image with a point drawn on it.
    """
    mask = np.array(
        Image.open(BytesIO(requests.get(
            tool["instanceURI"]).content)))[:, :, :3]
    mask[:, :, 1] *= 0
    mask[:, :, 2] *= 0
    weighted = cv2.addWeighted(image, alpha, mask, 1 - alpha, 0)
    image[np.sum(mask, axis=-1) > 0] = weighted[np.sum(mask, axis=-1) > 0]
    return image

def addLabelLayer(mask, i, tool):
    alpha = 0.5
    newMask = np.array(
        Image.open(BytesIO(requests.get(
            tool["instanceURI"]).content)))[:, :, :3]
    print(f'newMask.shape: {newMask.shape}')
    mask1d = np.zeros((720, 1280, 1), dtype=np.uint8)
    print(f'mask1d.shape: {mask1d.shape}')
    for a in range(720):
        for b in range(1280):
                mask1d[a, b] = (i + 1) if newMask[a, b, 0] > 0 or newMask[a, b, 1] > 0 or newMask[a, b, 2] > 0 else 0
                mask[a, b] = mask1d[a, b]
    # newMask[:, :, 1] *= 0
    # newMask[:, :, 2] *= 0
    # weighted = cv2.addWeighted(newMask, alpha, newMask, 1 - alpha, 0)
    # newMask[np.sum(newMask, axis=-1) > 0] = weighted[np.sum(newMask, axis=-1) > 0]
    #mask[newMask > 0] = 1 * 1/(i + 1)
    return mask
    
    

PROJECT_ID = 'ckr52dod66ha90ycjdenld9vg'
client = Client(api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJja3I1MjBqYWg2cDNwMHlhZWFrN2IzNDVnIiwib3JnYW5pemF0aW9uSWQiOiJja3I1MjBqYTM2cDNvMHlhZWYxdGtiOGk5IiwiYXBpS2V5SWQiOiJja3JiNTF0eW4wNG95MHk2NmJyamkxa2NiIiwic2VjcmV0IjoiYjg3MmU3ZTMzMDBjNzMyY2IwMjIxODU5MjZkYmM1OWIiLCJpYXQiOjE2MjY3MzAwMTYsImV4cCI6MjI1Nzg4MjAxNn0.YGVPslsuE9ggO4fiAJ1dxaYqDzMX7lsKJWKKW5tTRDw')
project = client.get_project(project_id=PROJECT_ID)
export_url = project.export_labels()
exports = requests.get(export_url).json()
for i in range(2):
    mask = np.zeros((720, 1280, 1), dtype=np.uint8)
    image = np.array(
        Image.open(BytesIO(requests.get(exports[i]["Labeled Data"]).content)))
    print(type(exports[i]["Label"]["objects"]))
    for j in range(len(exports[i]["Label"]["objects"])):
        tool = exports[i]["Label"]["objects"][j]
        if "bbox" in tool:
            image = visualize_bbox(image, tool)
        elif "point" in tool:
            image = visualize_point(image, tool)
        elif "polygon" in tool:
            image = visualize_poly(image, tool)
        # elif "instanceURI" in tool and tool['title'] == 'Background':
        #     # All tools have instanceURI but the car was made with the segmentation tool
        #     image = visualize_mask(image, tool)
        elif "instanceURI" in tool and tool['title'] == 'Lane Line':
            # All tools have instanceURI but the car was made with the segmentation tool
            image = visualize_mask(image, tool)
        elif "instanceURI" in tool and tool['title'] == 'Right Lane':
            # All tools have instanceURI but the car was made with the segmentation tool
            image = visualize_mask(image, tool)
        elif "instanceURI" in tool and tool['title'] == 'Left Lane':
            # All tools have instanceURI but the car was made with the segmentation tool
            image = visualize_mask(image, tool)
        elif "instanceURI" in tool and tool['title'] == 'F1Tenth Car':
            # All tools have instanceURI but the car was made with the segmentation tool
            image = visualize_mask(image, tool)

        mask = addLabelLayer(mask, j, tool)

    maskTensor = torch.tensor(mask)
    print(maskTensor.shape)
    print(f'mask shape: {mask.shape}')
    # mask = Image.fromarray(mask[0])
    # mask.show()
    threshed = cv2.adaptiveThreshold(mask, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
    cv2.imshow(threshed)
    imgTensor = torch.tensor(image)
    image = Image.fromarray(image.astype(np.uint8))
    w, h = image.size
    image.resize((w // 4, h // 4))
    print(imgTensor.shape)
    image.show()




