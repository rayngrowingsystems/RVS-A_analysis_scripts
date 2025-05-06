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


def execute(script_name, settings, mask_file_name, preview=False):  # this is the analysis workflow
    print("--> Execute:", script_name, settings)

    return_list = []

    # Load parameters from the settings dict TODO: Improve settings handling (using a class)
    # files and folder
    out_folder = settings["outputFolder"]

    # ROIs
    roi_items = settings["experimentSettings"]["roiInfo"]["roiList"]
    roi_mode_selection = settings["experimentSettings"]["roiInfo"]["detectionMode"]
    roi_mode_types = ["partial", "cutto", "largest"]  # types available for plantcv.roi.filter
    roi_mode = roi_mode_types[roi_mode_selection]

    # Crop rectangle
    crop_rectangle = settings["experimentSettings"]["cropRect"]

    # script specific settings (options are defined in the .config file)
    script_options = settings["experimentSettings"]["analysis"]["scriptOptions"]["general"]

    roi_overlay = script_options["roi_overlay"]
    mark_objects = script_options["mark_objects"]
    line_width = script_options["line_width"]

    # get temporary session information
    if "temporary" in settings["experimentSettings"]["sessionData"]:
        session_data = settings["experimentSettings"]["sessionData"]
        temp_data = session_data["temporary"]
        print("Loaded temporary session Data")
    else:
        session_data = dict()
        temp_data = dict()
        print("Created temporary session Data")

    # set plantcv variables
    pcv.params.line_thickness = int(line_width)
    pcv.params.debug = None

    # ANALYSIS WORKFLOW START
    print("--> Starting workflow")

    # determine the mask script based on the chosen option
    create_function = _get_mask_function(mask_file_name)

    # retrieving preprocessed data cube, meta data and mask
    spectral_array, rvs_metadata, mask = create_function(settings, mask_preview=False)

    # extract image name
    filename = spectral_array.filename  # TODO move this to rvs_metadata
    image_name = os.path.split(filename)[-1]
    image_name = os.path.splitext(image_name)[0]

    # copy unaltered pseudo rgb image for plotting results/debug information on it later
    img_labelled = np.copy(spectral_array.pseudo_rgb)

    if roi_items:  # only if ROIs are set
        # process ROI items forwarded from the UI
        rois = process_rois(roi_items, img_labelled, crop_rectangle)
    else:  # if no ROIs are set, no ROI filter is applied
        warnings.warn("No ROIs set and nothing is analyzed. Please set ROIs to perform the analysis!")
        return None

    # ANALYSES
    # plant detection start
    data_file_name = os.path.normpath(f"{out_folder['data']}/plant_detection.csv")
    num_rois = len(rois.contours)

    if "previous_detection" in temp_data:
        previous_detection = temp_data["previous_detection"]
        if len(previous_detection) != num_rois:
            print("Number of ROIs in previous run does not match the number of ROIs in the current run!")
    else:
        previous_detection = [False] * num_rois

    img_date_time = f"{rvs_metadata['capture date']} {rvs_metadata['capture time']}"

    for i, roi in enumerate(rois):
        kept_mask = pcv.roi.filter(mask, roi, roi_type=roi_mode)
        contours, _ = cv2.findContours(kept_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0 and previous_detection[i] is False:
            print(f"Object detected in ROI {i} ({img_date_time})")
            previous_detection[i] = img_date_time
        else:
            pass

        if mark_objects and len(contours) > 0:
            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Draw a yellow cross at each centroid
                    cross_color = (255, 255, 0)
                    cross_size = 20
                    cv2.drawMarker(img_labelled, (cx, cy), cross_color, markerType=cv2.MARKER_CROSS,
                                   markerSize=cross_size, thickness=line_width)

        # Choose color: Green if plant detected, Red otherwise
        color = (0, 255, 0) if previous_detection[i] else (0, 0, 255)
        draw_roi_overlay(img_labelled, roi, color)

    pseudo_rgb_file_name = os.path.normpath(f"{out_folder['images']}/{image_name}_pseudoRGB.png")

    print("Writing image to " + pseudo_rgb_file_name)
    pcv.print_image(img=img_labelled, filename=pseudo_rgb_file_name)
    return_list.append(
        (
            "preview",
            pseudo_rgb_file_name,
        )
    )

    print("Writing detection data")
    with open(data_file_name, "w") as f:
        f.write("ROI,timestamp\n")
        for i, timestamp in enumerate(previous_detection):
            if timestamp:
                f.write(f"{i+1},{timestamp}\n")
            else:
                f.write(f"{i + 1},No emergence detected\n")

    print("Writing session data")
    temp_data["previous_detection"] = previous_detection
    session_data["temporary"] = temp_data

    return_list.append(
        (
            "session_data",
            session_data,
        )
    )

    if preview:
        return pseudo_rgb_file_name

    print("--> Workflow done")

    # ANALYSIS WORKFLOW END

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


def initialize_detection_table(roi_count, existing_file=None):
    """
    Create or load a detection table to track ROI emergence over time.
    Columns:
        - ROI index
        - Emergence time (timestamp string or False)
        - Center activity seen (bool)
        - Outer activity seen (bool)
        - Potential intrusion (bool)
    """
    if existing_file and os.path.exists(existing_file):
        df = pd.read_csv(existing_file)
        df.set_index("ROI", inplace=True)
    else:
        df = pd.DataFrame({
            "ROI": list(range(roi_count)),
            "emergence_time": [False] * roi_count,
            "center_activity": [False] * roi_count,
            "outer_activity": [False] * roi_count,
            "potential_intrusion": [False] * roi_count,
        }).set_index("ROI")

    return df


def update_detection_table(df, roi_index, timestamp, center, outer, intrusion=False):
    """
    Update the detection dataframe for a given ROI.
    """
    if center:
        df.at[roi_index, "center_activity"] = True
    if outer:
        df.at[roi_index, "outer_activity"] = True
    if intrusion:
        df.at[roi_index, "potential_intrusion"] = True
    if df.at[roi_index, "emergence_time"] is False and not intrusion:
        df.at[roi_index, "emergence_time"] = timestamp
    return df


def save_detection_table(df, path):
    df.to_csv(path)


def is_likely_intrusion(cnt, roi_contour, roi_record, area_threshold=100, safe_zone_ratio=0.25):
    """
    Determine if a contour is a likely intrusion based on ROI activity history.

    Parameters:
    - cnt: Detected object contour
    - roi_contour: Numpy array of the ROI shape
    - roi_record: One row (Series) from the detection DataFrame for the current ROI
    - area_threshold: Area threshold for detecting large objects
    - safe_zone_ratio: Fraction of ROI radius considered "central"

    Returns:
    - True if likely intrusion, False otherwise
    """

    area = cv2.contourArea(cnt)
    if area < 1:
        return False

    M = cv2.moments(cnt)
    if M["m00"] == 0:
        return False

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    x, y, w, h = cv2.boundingRect(roi_contour)
    center_x = x + w // 2
    center_y = y + h // 2
    safe_radius = int(min(w, h) * safe_zone_ratio)

    dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
    is_center = dist < safe_radius
    is_outer = not is_center

    # Read historical flags from the ROI record
    center_seen = roi_record["center_activity"]
    outer_seen = roi_record["outer_activity"]

    if is_center:
        return False  # not an intrusion
    if is_outer and area > area_threshold and outer_seen and not center_seen:
        return True

    return False
