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


# Default mask workflow. Selection of other mask scripts is possible in the UI.
def create_mask(settings, mask_preview=True):

    # extract masking setting, available options are defined in the .conf file
    mask_options = settings["experimentSettings"]["analysis"]["maskOptions"]

    selected_wl = mask_options["wavelength"]
    wl_thresh = mask_options["wl_thresh"]
    fill_size = mask_options["fill_size"]
    dilate_pixel = mask_options["dilate_pixel"]

    spectral_array, rvs_metadata = rayn_utils.prepare_spectral_data(settings)

    # get data from selected wavelength band
    if (selected_wl != "None") and (selected_wl != ""):
        selected_layer = spectral_array.array_data[:, :, int(spectral_array.wavelength_dict[int(selected_wl)])]
    else:
        selected_layer = spectral_array.array_data[:, :, 0]
        warnings.warn("No wavelength for mask selected. Defaulting to first in list")

    # create binary mask from layer using an adjustable threshold
    binary_img = pcv.threshold.binary(gray_img=selected_layer, threshold=wl_thresh)
    binary_img = pcv.fill(bin_img=binary_img, size=fill_size)

    if dilate_pixel:
        binary_img = pcv.dilate(gray_img=binary_img, ksize=2, i=2)

    # creates mask preview image
    create_mask_preview(binary_img, settings, mask_preview)

    return spectral_array, rvs_metadata, binary_img


def execute(feedback_queue, script_name, settings, mask_file_name):  # this is the analysis workflow
    print("Execute:", script_name, settings)

    # Load parameters from the settings dict
    # files and folder
    out_folder = settings["outputFolder"]

    # ROIs
    roi_items = settings["experimentSettings"]["roiInfo"]["roiList"]
    roi_mode_selection = settings["experimentSettings"]["roiInfo"]["detectionMode"]
    roi_mode_types = ["partial", "cutto", "largest"]  # types available for plantcv.roi.filter
    roi_mode = roi_mode_types[roi_mode_selection]

    # script specific settings (options are defined in the .config file)
    script_options = settings["experimentSettings"]["analysis"]["scriptOptions"]["general"]

    selected_index = script_options["index_selection"]
    roi_overlay = script_options["roi_overlay"]
    line_width = script_options["line_width"]

    # script specific settings for charting (options are defined in the .config file)

    # set plantcv variables
    pcv.params.line_thickness = int(line_width)
    pcv.params.debug = None

    # determine mask script based on the chosen option
    if mask_file_name != "":  # external mask script (= mask function defined in another file)
        mask_path, mask_file = os.path.split(mask_file_name)
        print("External mask file used: ", mask_file_name)

        sys.path.append(mask_path)
        mask_script = importlib.import_module(mask_file.replace(".py", ""))
        create_function = mask_script.create_mask

    else:  # default/internal mask script is used (= mask function defined in this script)
        print("Internal mask used")
        
        create_function = create_mask

    # ANALYSIS WORKFLOW START
    print("Starting workflow")

    # retrieving preprocessed data cube and mask from another script
    spectral_array, rvs_metadata, mask = create_function(settings, mask_preview=False)

    # extract image name
    filename = spectral_array.filename
    image_name = os.path.split(filename)[-1]
    image_name = os.path.splitext(image_name)[0]

    # signal which file is processed
    feedback_queue.put([script_name, 'Processing: ' + spectral_array.filename])

    # copy unaltered pseudo rgb image for plotting results/debug information on it later
    img_labelled = np.copy(spectral_array.pseudo_rgb)

    if roi_items:  # only if ROIs are set
        # process ROI items forwarded from the UI
        rois = process_rois(roi_items, img_labelled, roi_debug=roi_overlay)
        # identify objects in the ROIs
        labeled_objects, n_obj = pcv.create_labels(mask=mask, rois=rois, roi_type=roi_mode)

    else:  # if no ROIs are set, no ROI filter is applied
        labeled_objects, n_obj = pcv.create_labels(mask=mask, rois=None)

    # ANALYSES
    # analyze shape
    img_labelled = pcv.analyze.size(img=img_labelled, labeled_mask=labeled_objects, n_labels=n_obj, label="plant")

    # analyze spectral reflectance
    spectral_hist = pcv.analyze.spectral_reflectance(hsi=spectral_array, labeled_mask=labeled_objects, n_labels=n_obj,
                                                     label="plant")

    # analyze reflectance index
    index_functions = rayn_utils.get_index_functions()  # load all available index functions
    index_array = index_functions[selected_index][1](spectral_array, 10)  # call the function of the selected index
    index_hist = pcv.analyze.spectral_index(index_img=index_array, labeled_mask=labeled_objects, n_labels=n_obj,
                                            label="plant")

    # create pseudocolor representation
    index_pseudocolor = pcv.visualize.pseudocolor(gray_img=index_array.array_data, mask=mask,
                                                  background="white", axes=False,
                                                  colorbar=False, cmap='viridis',
                                                  min_value=index_functions[selected_index][2],
                                                  max_value=index_functions[selected_index][3])

    # return preview image and
    pseudo_rgb_file_name = os.path.normpath(f"{out_folder}/ProcessedImages/{image_name}_pseudoRGB.png")
    spectral_hist_file_name = os.path.normpath(f"{out_folder}/VisualResults/{image_name}_spectral_histogram.png")
    index_hist_file_name = os.path.normpath(f"{out_folder}/VisualResults/{image_name}_index_histogram.png")
    index_pseudocolor_file_name = os.path.normpath(f"{out_folder}/VisualResults/{image_name}_index_pseudocolor.png")

    path1, file_name = os.path.split(pseudo_rgb_file_name)
    path2, file_name = os.path.split(spectral_hist_file_name)

    if not os.path.exists(path1):
        os.makedirs(path1)
        print("Created folder " + path1)

    if not os.path.exists(path2):
        os.makedirs(path2)
        print("Created folder " + path2)

    print("Writing image to " + pseudo_rgb_file_name)
    pcv.print_image(img=img_labelled, filename=pseudo_rgb_file_name)

    print("Writing visual outputs to " + path2)
    pcv.print_image(img=spectral_hist, filename=spectral_hist_file_name)
    pcv.print_image(img=index_hist, filename=index_hist_file_name)
    pcv.print_image(img=index_pseudocolor, filename=index_pseudocolor_file_name)

    # Use feedbackQueue.put to send feedback to the main application
    # feedbackQueue.put([name, 'Processing images...'])
    print("Writing info to queue")
    feedback_queue.put([script_name, 'preview', pseudo_rgb_file_name])
    feedback_queue.put([script_name, 'spectral_hist', spectral_hist_file_name])
    feedback_queue.put([script_name, 'index_hist', index_hist_file_name])
    feedback_queue.put([script_name, 'index_pseudocolor', index_pseudocolor_file_name])

    print("Workflow done")

    # ANALYSIS WORKFLOW END

    # adding meta data to outputs
    pcv.outputs.add_metadata("camera", str, rvs_metadata["camera"])
    pcv.outputs.add_metadata("firmware", str, rvs_metadata["firmware version"])
    pcv.outputs.add_metadata("timestamp", datetime.datetime,
                             f"{rvs_metadata['capture date']} {rvs_metadata['capture time']}")
    pcv.outputs.add_metadata("pixel_to_mm_factor", datetime.date, rvs_metadata["px to mm ratio"])

    data_file_name = os.path.normpath(out_folder + "/RawData/" + image_name + ".json")
    path, file_name = os.path.split(data_file_name)

    if not os.path.exists(path):
        os.makedirs(path)
        print("Created folder " + path)
    print("Writing raw data to " + data_file_name)

    pcv.outputs.save_results(data_file_name, outformat="json")
    pcv.outputs.clear()

    # signal results file
    feedback_queue.put([script_name, 'results', data_file_name])


def get_display_name_for_chart(settings):

    # load settings
    script_options = settings["experimentSettings"]["analysis"]["scriptOptions"]["general"]

    analyze_index = script_options["analyze_index"]
    selected_index = script_options["index_selection"]
    analyze_shape = script_options["analyze_shape"]

    plot_selection = settings["experimentSettings"]["analysis"]["chartOptions"]["plot_selection"]

    title = ""
    y_label = ""

    if plot_selection == "plot_index" and analyze_index:
        index_dict_dd = rayn_utils.get_index_functions()
        full_index_name = index_dict_dd[selected_index][0]
        title = full_index_name
        y_label = "relative index value"

    if plot_selection in ["area", "width", "height", "perimeter"] and analyze_shape:
        title = f"Leaf {plot_selection}"
        y_label = f"Leaf {plot_selection} [px]"

    else:
        if analyze_shape:
            title = f"Leaf {plot_selection}"
            y_label = f"Leaf {plot_selection} [px]"

        if analyze_index:
            index_dict_dd = rayn_utils.get_index_functions()
            full_index_name = index_dict_dd[selected_index][0]
            title = full_index_name
            y_label = "relative index value"

    return title, y_label


def dropdown_values(setting, wavelengths):  # fills UI element with values
    if setting == "index_list":  # selects the respective UI element
        index_dict_dd = rayn_utils.get_index_functions()
        name_list = list(index_dict_dd)
        display_name_list = [item[0] for item in index_dict_dd.values()]

        return display_name_list, name_list

    else:
        return


def process_rois(roi_items, rgb_image, roi_debug=False):  # get the rois from individual coordinates
    # creating empty ROI object
    rois = pcv.Objects(contours=[], hierarchy=[])

    for item in roi_items:
        print("RoiItem:", item)

        roi_type = item["type"]
        roi_x = item["x"]
        roi_y = item["y"]
        roi_width = item["width"]
        roi_height = item["height"]

        if roi_type == "Circle":
            roi_radius = int(roi_width/2)
            # create a single circular ROI
            roi = pcv.roi.circle(x=roi_x, y=roi_y, r=roi_radius, img=rgb_image)
        elif roi_type == "Rectangle":
            # create a single rectangle ROI
            print("calculated x/y", roi_x - roi_width/2, roi_y - roi_height/2)
            roi = pcv.roi.rectangle(x=roi_x - roi_width/2, y=roi_y - roi_height/2,
                                    h=roi_height, w=roi_width, img=rgb_image)
        elif roi_type == "Ellipse":
            roi_radius1 = int(roi_width/2)
            roi_radius2 = int(roi_height/2)
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


def create_mask_preview(mask, settings, create_preview=True):
    if create_preview:
        out_image = settings["outputImage"]
        image_file_name = os.path.normpath(out_image)
        print("Writing image to " + image_file_name)
        pcv.print_image(img=mask, filename=image_file_name)
