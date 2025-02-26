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

import os
import numpy as np
import warnings
import datetime

from plantcv import plantcv as pcv
import rayn_utils
import sys
import importlib
import cv2

import vl_convert as vlc

import matplotlib
matplotlib.use('agg')


# Default mask workflow. Selection of other mask scripts is possible in the UI.
def create_mask(settings, mask_preview=True):
    # extract masking setting, available options are defined in the .conf file
    mask_options = settings["experimentSettings"]["analysis"]["maskOptions"]

    spectral_array, rvs_metadata = rayn_utils.prepare_spectral_data(settings, preview=mask_preview)

    # get data from selected wavelength band
    if (mask_options["wavelength"] != "None") and (mask_options["wavelength"] != ""):
        selected_layer = spectral_array.array_data[:, :, int(spectral_array.wavelength_dict[int(mask_options["wavelength"])])]
    else:
        selected_layer = spectral_array.array_data[:, :, 0]
        warnings.warn("No wavelength for mask selected. Defaulting to first in list")

    # create binary mask from layer using an adjustable threshold
    binary_img = pcv.threshold.binary(gray_img=selected_layer, threshold=mask_options["wl_thresh"])
    binary_img = pcv.fill(bin_img=binary_img, size=mask_options["fill_size"])

    if mask_options["dilate_pixel"]:
        binary_img = pcv.dilate(gray_img=binary_img, ksize=2, i=2)

    if mask_options["invert_mask"]:
        binary_img = pcv.invert(binary_img)

    # creates mask preview image
    rayn_utils.create_mask_preview(binary_img, spectral_array.pseudo_rgb, settings, mask_preview)

    return spectral_array, rvs_metadata, binary_img


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

    selected_index = script_options["index_selection"]

    roi_overlay = script_options["roi_overlay"]
    line_width = script_options["line_width"]
    #convert_pixel = script_options["convert_pixel"]

    false_color_image = script_options["false_color_image"]
    spectral_histogram = script_options["spectral_histogram"]
    index_histogram = script_options["index_histogram"]

    # set plantcv variables
    pcv.params.line_thickness = int(line_width)
    pcv.params.debug = None

    # ANALYSIS WORKFLOW START
    print("--> Starting workflow")

    # determine mask script based on the chosen option
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
        rois = process_rois(roi_items, img_labelled, crop_rectangle, roi_debug=roi_overlay)
        # identify objects in the ROIs
        labeled_objects, n_obj = pcv.create_labels(mask=mask, rois=rois, roi_type=roi_mode)

    else:  # if no ROIs are set, no ROI filter is applied
        labeled_objects, n_obj = pcv.create_labels(mask=mask, rois=None)

    # ANALYSES
    # analyze shape
    img_labelled = pcv.analyze.size(img=img_labelled, labeled_mask=labeled_objects, n_labels=n_obj, label="plant")
    pseudo_rgb_file_name = os.path.normpath(f"{out_folder['images']}/{image_name}_pseudoRGB.png")

    print("Writing image to " + pseudo_rgb_file_name)
    pcv.print_image(img=img_labelled, filename=pseudo_rgb_file_name)
    return_list.append(("preview", pseudo_rgb_file_name,))

    if preview:
        return pseudo_rgb_file_name

    # analyze spectral reflectance
    spectral_hist = pcv.analyze.spectral_reflectance(hsi=spectral_array, labeled_mask=labeled_objects, n_labels=n_obj,
                                                     label="plant")

    # analyze reflectance index
    index_functions = rayn_utils.get_index_functions()  # load all available index functions

    index_results = {}
    if selected_index:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for index in selected_index:
                index_array = index_functions[index][1](spectral_array, 10)  # call the function of the selected index
                index_hist = pcv.analyze.spectral_index(index_img=index_array, labeled_mask=labeled_objects,
                                                        n_labels=n_obj, label="plant")
                index_results[index] = (index_array, index_hist)

    # return visual results
    print("Writing visual outputs to " + out_folder['visuals'])
    if spectral_histogram:
        spectral_hist_file_name = os.path.normpath(f"{out_folder['visuals']}/{image_name}_spectral_histogram.png")
        chart_dict = spectral_hist.to_dict()
        chart_dict["spec"]["mark"]["point"] = True
        png_data = vlc.vegalite_to_png(chart_dict, scale=1.5)
        with open(spectral_hist_file_name, "wb") as f:
            f.write(png_data)

        return_list.append(("spectral_hist", spectral_hist_file_name,))

    if index_histogram:
        for index, results_data in index_results.items():
            index_hist_file_name = os.path.normpath(f"{out_folder['visuals']}/{image_name}_{index}_histogram.png")
            chart_dict = results_data[1].to_dict()
            png_data = vlc.vegalite_to_png(chart_dict, scale=1.5)
            with open(index_hist_file_name, "wb") as f:
                f.write(png_data)

            return_list.append((f"index_hist_{index}", index_hist_file_name,))

    if false_color_image:
        # create false color representation
        for index, results_data in index_results.items():
            object_mask = np.where(labeled_objects > 0, 1, 0)
            # masked_array = np.ma.array(results_data[0].array_data, mask=(object_mask > 0))
            # masked_array = np.ma.masked_invalid(masked_array)

            index_false_color = pcv.visualize.pseudocolor(gray_img=results_data[0].array_data, mask=object_mask,
                                                          background="white", axes=False,
                                                          colorbar=True, cmap='viridis',
                                                          min_value=index_functions[index][2],
                                                          max_value=index_functions[index][3])

            index_false_color_file_name = os.path.normpath(f"{out_folder['visuals']}/{image_name}_{index}_false_color.png")
            pcv.print_image(img=index_false_color, filename=index_false_color_file_name)
            return_list.append((f"index_false_color_{index}", index_false_color_file_name,))

    print("--> Workflow done")

    # ANALYSIS WORKFLOW END

    # adding meta data to outputs
    pcv.outputs.add_metadata("camera", str, rvs_metadata["camera"])
    pcv.outputs.add_metadata("firmware", str, rvs_metadata["firmware version"])
    pcv.outputs.add_metadata("timestamp", datetime.datetime,
                             f"{rvs_metadata['capture date']} {rvs_metadata['capture time']}")
    pcv.outputs.add_metadata("filename", str, image_name)
    pcv.outputs.add_metadata("pixel_to_mm_factor", datetime.date, rvs_metadata["px to mm ratio"])

    data_file_name = os.path.normpath(f"{out_folder['data']}/{image_name}.json")

    print("--> Writing raw data to " + data_file_name)

    pcv.outputs.save_results(data_file_name, outformat="json")
    pcv.outputs.clear()

    # signal results file
    return_list.append(("results", data_file_name,))
    # feedback_queue.put([script_name, 'results', data_file_name])

    return return_list


def dropdown_values(setting, wavelengths):  # fills UI element with values
    if setting == "index_list":  # selects the respective UI element
        index_dict_dd = rayn_utils.get_index_functions()
        name_list = list(index_dict_dd)
        display_name_list = [item[0] for item in index_dict_dd.values()]

        return display_name_list, name_list

    else:
        return


def process_rois(roi_items, rgb_image, crop_rectangle, roi_debug=False):  # get the rois from individual coordinates
    # creating empty ROI object
    rois = pcv.Objects(contours=[], hierarchy=[])

    for item in roi_items:
        print("RoiItem:", item)

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
            print("calculated x/y", roi_x - roi_width / 2, roi_y - roi_height / 2)
            roi = pcv.roi.rectangle(x=roi_x - roi_width / 2, y=roi_y - roi_height / 2,
                                    h=roi_height, w=roi_width, img=rgb_image)
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


def draw_roi_overlay(img, roi_contour):
    color = (255, 0, 0)

    for i, cnt in enumerate(roi_contour):
        cv2.drawContours(img, cnt.contours[0], -1, color, pcv.params.line_thickness)


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
