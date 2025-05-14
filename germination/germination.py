# Copyright 2024 ETC Inc d/b/a RAYN Growing Systems
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import importlib
import os
import sys
import warnings

import cv2
import matplotlib
import numpy as np
import pandas as pd

import rayn_utils
from plantcv import plantcv as pcv

matplotlib.use("agg")


# Default mask workflow. Selection of other mask scripts is possible in the UI.
def create_mask(settings, mask_preview=True):
    # extract masking setting, available options are defined in the .conf file
    mask_options = settings["experimentSettings"]["analysis"]["maskOptions"]

    spectral_array, rvs_metadata = rayn_utils.prepare_spectral_data(settings, preview=mask_preview)

    # get data from the selected wavelength band
    if (mask_options["wavelength"] != "None") and (mask_options["wavelength"] != ""):
        selected_layer = spectral_array.array_data[
                         :, :, int(spectral_array.wavelength_dict[int(mask_options["wavelength"])])
                         ]
    else:
        selected_layer = spectral_array.array_data[:, :, 0]
        warnings.warn("No wavelength for mask selected. Defaulting to first in list")

    # create binary mask from layer using an adjustable threshold
    wl_mask = pcv.threshold.binary(gray_img=selected_layer, threshold=mask_options["wl_thresh"])
    wl_mask = pcv.fill(bin_img=wl_mask, size=mask_options["fill_wl"])

    # create binary mask from index using an adjustable threshold
    ari = pcv.spectral_index.ari(hsi=spectral_array, distance=20)
    ari_mask = pcv.threshold.binary(gray_img=ari.array_data, threshold=mask_options["ari_thresh"], object_type="dark")
    ari_mask = pcv.fill(ari_mask, size=mask_options["fill_ari"])

    if mask_options["dilate_pixel"]:
        wl_mask = pcv.dilate(gray_img=wl_mask, ksize=3, i=1)

    final_mask = pcv.logical_and(ari_mask, wl_mask)
    final_mask = pcv.fill(final_mask, size=mask_options["fill_combined"])

    mask_dict = {
        "ari_mask": ari_mask,
        "wl_mask": wl_mask,
        "final_mask": final_mask
    }

    rayn_utils.create_mask_preview(mask_dict[mask_options["show_mask"]],
                                   spectral_array.pseudo_rgb,
                                   settings,
                                   mask_preview)

    return spectral_array, rvs_metadata, final_mask


def execute(script_name, settings, mask_file_name, preview=False):
    print("--> Execute:", script_name, settings)

    return_list = []
    out_folder = settings["outputFolder"]

    # ROIs and crop settings
    roi_items = settings["experimentSettings"]["roiInfo"]["roiList"]
    crop_rectangle = settings["experimentSettings"]["cropRect"]

    # Script options
    script_options = settings["experimentSettings"]["analysis"]["scriptOptions"]["general"]
    roi_overlay = script_options["roi_overlay"]
    mark_objects = script_options["mark_objects"]
    line_width = script_options["line_width"]

    # Session handling
    session_data = settings["experimentSettings"].get("sessionData", {})
    temp_data = session_data.get("temporary", {})
    print("Loaded temporary session Data" if temp_data else "Created temporary session Data")

    pcv.params.line_thickness = int(line_width)
    pcv.params.debug = None

    # Load image and mask
    create_function = _get_mask_function(mask_file_name)
    spectral_array, rvs_metadata, mask = create_function(settings, mask_preview=False)
    image_name = os.path.splitext(os.path.split(spectral_array.filename)[-1])[0]
    img_labelled = np.copy(spectral_array.pseudo_rgb)

    if not roi_items:
        warnings.warn("No ROIs set. Analysis skipped.")
        return None

    rois = process_rois(roi_items, img_labelled, crop_rectangle)
    num_rois = len(rois.contours)

    # Init or load detection info and centroid histories
    if "detection_info" in temp_data:
        detection_info = pd.DataFrame(temp_data["detection_info"])
        print("Loaded previous detection_info from session")
    else:
        detection_info = pd.DataFrame({
            "ROI": list(range(num_rois)),
            "emergence_time": [False] * num_rois,
            "center_activity": [False] * num_rois,
            "outer_activity": [False] * num_rois,
            "potential_intrusion": [False] * num_rois,
        }).set_index("ROI")

    centroid_history = temp_data.get("centroid_history", {i: [] for i in range(num_rois)})
    outer_centroid_history = temp_data.get("outer_centroid_history", {i: [] for i in range(num_rois)})

    # Colors
    COLOR_GREEN  = (0, 255, 0)
    COLOR_RED    = (0, 0, 255)
    COLOR_ORANGE = (0, 165, 255)
    COLOR_YELLOW = (0, 255, 255)

    img_date_time = f"{rvs_metadata['capture date']} {rvs_metadata['capture time']}"
    date = datetime.datetime.strptime(img_date_time, "%Y-%m-%d %H:%M:%S")
    vis_img = img_labelled

    print(f"Analyzing image taken {img_date_time}")
    for i, roi in enumerate(rois):
        if detection_info.at[i, "emergence_time"] != False:
            print(f"skipping ROI {i}")
            continue

        buffer_roi = create_buffer_zone_roi(vis_img, roi, 0.1)

        kept_mask = pcv.roi.filter(mask, roi, roi_type='partial')
        kept_buffer_mask = pcv.roi.filter(mask, buffer_roi, roi_type='partial')

        contours, _ = cv2.findContours(kept_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        buffer_contours, _ = cv2.findContours(kept_buffer_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        object_in_roi = len(contours) > 0
        object_in_buffer = len(buffer_contours) > 0

        cross_size = 30

        if object_in_roi:
            centroids_in_roi = []
            for cnt in contours:
                center = is_object_in_center(cnt, roi, threshold_ratio=0.5)

                cx, cy = get_object_centroid(cnt)
                centroids_in_roi.append((cx, cy))
                centroid_history[i].append(centroids_in_roi)

                cross_color = COLOR_ORANGE

                if center:
                    if detection_info.at[i, "potential_intrusion"] and len(centroid_history[i]) >= 2:
                        # Get previous centroids (from last timepoint)
                        previous_centroids = centroid_history[i][-2]
                        current_centroids = centroid_history[i][-1]
                        print(
                            f"Center object in ROI {i} detected! Checking if this is just the previously detected intruder...")

                        if are_centroids_close(current_centroids, previous_centroids, max_distance=30):
                            cross_color = COLOR_RED
                            print(f"⚠️ Center object in ROI {i} is likely continuation of previous intrusion.")
                            # keep potential_intrusion flag, skip emergence_time
                        else:
                            cross_color = COLOR_GREEN
                            print(f"🌱 New valid center object in ROI {i}")
                            detection_info.at[i, "emergence_time"] = img_date_time
                    else:
                        cross_color = COLOR_GREEN
                        print(f"🌱 Center object in ROI {i} ({img_date_time})")
                        detection_info.at[i, "emergence_time"] = img_date_time

                elif not center and detection_info.at[i, "outer_activity"] == False:
                    cross_color = COLOR_GREEN
                    print(f"🔵 Edge object in ROI {i}, no prior surrounding activity ({img_date_time})")
                    detection_info.at[i, "emergence_time"] = img_date_time
                    # you could also track secondary_centroid here if needed

                elif not center and detection_info.at[i, "outer_activity"] != False:
                    print(
                        f"⚠️ Edge object in ROI {i}, with prior surroundings activity — likely intrusion, looking deeper ... ({img_date_time})")

                    previous_outer = outer_centroid_history[i][-1]
                    current_centroids = centroid_history[i][-1]

                    if are_centroids_close(current_centroids, previous_outer, max_distance=30):
                        print(f"⚠️ Edge object in ROI {i} close to previous object in buffer — likely intrusion")
                        if not detection_info.at[i, "potential_intrusion"]:
                            detection_info.at[i, "potential_intrusion"] = img_date_time
                        cross_color = COLOR_YELLOW
                    elif detection_info.at[i, "potential_intrusion"] and are_centroids_close(current_centroids,
                                                                                             centroid_history[i][-2],
                                                                                             max_distance=30):
                        print(
                            f"⚠️ Edge object in ROI {i} close to previous edge object — likely continuation of previous intrusion.")
                        cross_color = COLOR_RED
                    else:
                        print(f"New distant edge object — seems to be an emergence")
                        print(previous_outer, current_centroids)
                        detection_info.at[i, "emergence_time"] = img_date_time

                cv2.drawMarker(vis_img, (cx, cy), cross_color, markerType=cv2.MARKER_CROSS,
                               markerSize=cross_size, thickness=2)

        if object_in_buffer and object_in_roi == False:
            print(f"Outer acitivity detected at ROI {i}")
            detection_info.at[i, "outer_activity"] = img_date_time
            centroids_in_buffer = []

            for cnt in buffer_contours:
                cx, cy = get_object_centroid(cnt)
                centroids_in_buffer.append((cx, cy))
                outer_centroid_history[i].append(centroids_in_buffer)

        if detection_info.at[i, "emergence_time"]:
            color = COLOR_GREEN
        elif detection_info.at[i, "potential_intrusion"]:
            color = COLOR_YELLOW
        elif detection_info.at[i, "outer_activity"]:
            color = COLOR_ORANGE
        else:
            color = COLOR_RED

        # Draw the contour in the selected color
        cv2.drawContours(vis_img, roi.contours[0][0], -1, color, pcv.params.line_thickness)

    # Update session
    temp_data["detection_info"] = detection_info.to_dict()
    temp_data["centroid_history"] = centroid_history
    temp_data["outer_centroid_history"] = outer_centroid_history
    session_data["temporary"] = temp_data
    return_list.append(("session_data", session_data))

    pseudo_rgb_file_name = os.path.normpath(f"{out_folder['images']}/{image_name}_pseudoRGB.png")
    print("Writing image to " + pseudo_rgb_file_name)
    pcv.print_image(img=img_labelled, filename=pseudo_rgb_file_name)
    return_list.append(
        (
            "preview",
            pseudo_rgb_file_name,
        )
    )

    print("--> Workflow done")
    return return_list



def dropdown_values(setting, wavelengths):  # fills UI element with values
    if setting == "index_list":  # selects the respective UI element
        index_dict_dd = rayn_utils.get_index_functions()
        name_list = list(index_dict_dd)
        display_name_list = [item[0] for item in index_dict_dd.values()]

        return display_name_list, name_list

    if setting == "mask_list":  # defines the UI element this is applied to
        mask_dict = {
            "ari_mask": "ARI inverted mask",
            "wl_mask": "Wavelength mask",
            "final_mask": "Combined Mask"
        }

        name_list = list(mask_dict)
        display_name_list = [item for item in mask_dict.values()]

        return display_name_list, name_list

    else:
        return


def process_rois(roi_items, rgb_image, crop_rectangle, roi_debug=False):  # get the rois from individual coordinates
    # creating empty ROI objects
    rois = pcv.Objects(contours=[], hierarchy=[])

    for item in roi_items:
        roi_type = item["type"]
        roi_x = item["x"] - crop_rectangle[0]
        roi_y = item["y"] - crop_rectangle[1]
        roi_width = item["width"]
        roi_height = item["height"]

        if roi_type == "Circle":
            roi_radius = int(roi_width / 2)
            # create a single circular ROI
            roi = pcv.roi.circle(x=roi_x, y=roi_y, r=roi_radius, img=rgb_image)
        elif roi_type == "Rectangle":
            # create a single rectangle ROI
            roi = pcv.roi.rectangle(
                x=roi_x - roi_width / 2, y=roi_y - roi_height / 2, h=roi_height, w=roi_width, img=rgb_image
            )
        elif roi_type == "Ellipse":
            roi_radius1 = int(roi_width / 2)
            roi_radius2 = int(roi_height / 2)
            # create a single elliptical ROI
            roi = pcv.roi.ellipse(x=roi_x, y=roi_y, r1=roi_radius1, r2=roi_radius2, img=rgb_image, angle=0)
        elif roi_type == "Polygon":
            roi = pcv.roi.custom(vertices=item["points"], img=rgb_image)
        else:
            warnings.warn(f"ROI type {roi_type} not valid")
            break

        # append the roi contour and hierarchy to the object collecting all the rois
        rois.append(roi.contours, roi.hierarchy)

        if roi_debug:
            draw_roi_overlay(rgb_image, roi)

    return rois


def draw_roi_overlay(img, roi_contour, color=None):
    if color is None:
        color = (255, 0, 0)

    for cnt in roi_contour.contours:
        cv2.drawContours(img, cnt[0], -1, color, pcv.params.line_thickness)


def _get_mask_function(mask_script_filename):
    if mask_script_filename != "":  # external mask script (= mask function defined in another file)
        mask_path, mask_file = os.path.split(mask_script_filename)
        print("External mask file used: ", mask_script_filename)

        sys.path.append(mask_path)
        mask_script = importlib.import_module(mask_file.replace(".py", ""))

        return mask_script.create_mask

    else:  # default/internal mask script is used (= mask function defined in this script)
        print("Internal mask used")

        return create_mask


def unwrap_contour(roi):
    """Extract the innermost NumPy contour array from a nested PlantCV ROI object."""
    contour = roi.contours[0]

    # Keep unwrapping lists or tuples
    while isinstance(contour, (list, tuple)):
        contour = contour[0]

    # Verify it's a valid OpenCV contour: shape (N, 1, 2)
    if isinstance(contour, np.ndarray) and contour.ndim == 3 and contour.shape[1:] == (1, 2):
        return contour
    else:
        raise ValueError(
            f"Invalid contour shape or type: {type(contour)}, shape: {getattr(contour, 'shape', 'unknown')}")


def get_roi_centroid(roi):
    contour = unwrap_contour(roi)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return (cx, cy)
    return None


def estimate_radius_from_contour(roi, centroid):
    contour = unwrap_contour(roi)
    cx, cy = centroid
    distances = np.sqrt((contour[:, 0, 0] - cx) ** 2 + (contour[:, 0, 1] - cy) ** 2)
    return int(np.max(distances))  # use max to ensure safe zone fully wraps


def create_buffer_zone_roi(img, roi, buffer_zone_ratio):
    centroid = get_roi_centroid(roi)
    radius = estimate_radius_from_contour(roi, centroid)
    buffer_radius = int(radius + radius * buffer_zone_ratio)
    roi = pcv.roi.circle(img=img, x=centroid[0], y=centroid[1], r=buffer_radius)

    return roi


def is_object_in_center(detected_object, roi, threshold_ratio=0.5):
    roi_centroid = get_roi_centroid(roi)
    if roi_centroid is None:
        return None

    roi_radius = estimate_radius_from_contour(roi, roi_centroid)

    M = cv2.moments(detected_object)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    dist = np.sqrt((cx - roi_centroid[0]) ** 2 +
                   (cy - roi_centroid[1]) ** 2)

    return (cx, cy) if dist < roi_radius * threshold_ratio else None


def get_object_centroid(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        # Fallback: use geometric center (mean of all points)
        cx = int(np.mean(contour[:, 0, 0]))
        cy = int(np.mean(contour[:, 0, 1]))
        print(f"[fallback] m00 == 0, using mean centroid: ({cx}, {cy})")

    return cx, cy


from scipy.spatial.distance import cdist


def are_centroids_close(current, previous, max_distance=30):
    if not current or not previous:
        return False
    distances = cdist(current, previous)
    return np.any(distances < max_distance)