# Copyright 2024 RAYN Growing Systems
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
from plantcv import plantcv as pcv
import rayn_utils
import sys
import importlib


# Default mask workflow. Selection of other mask scripts is possible in the UI.
def create_mask(settings, mask_preview=True):

    # extract masking setting, available options are defined in the .conf file
    mask_options = settings["experimentSettings"]["analysis"]["maskOptions"]

    selected_wl = mask_options["wavelength"]
    wl_thresh = mask_options["wl_thresh"]
    fill_size = mask_options["fill_size"]
    dilate_pixel = mask_options["dilate_pixel"]

    spectral_array = prepare_spectral_data(settings)

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

    return spectral_array, binary_img


def execute(feedback_queue, script_name, settings, mask_file_name):  # this is the analysis workflow
    print("Execute:", script_name, settings)

    # Load parameters from the settings dict
    # files and folder
    out_folder = settings["outputFolder"]

    # ROIs
    roi_items = settings["experimentSettings"]["roiInfo"]["roiItems"]

    # script specific settings (options are defined in the .config file)
    script_options = settings["experimentSettings"]["analysis"]["scriptOptions"]["general"]

    analyze_index = script_options["analyze_index"]
    selected_index = script_options["index_selection"]
    analyze_shape = script_options["analyze_shape"]
    line_width = script_options["line_width"]

    # script specific settings for charting (options are defined in the .config file)
    plot_selection = settings["experimentSettings"]["analysis"]["chartOptions"]["plot_selection"]

    # set plantcv variables
    pcv.params.line_thickness = line_width
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
    spectral_array, mask = create_function(settings, mask_preview=False)

    # extract image name
    filename = spectral_array.filename
    image_name = os.path.split(filename)[-1]
    image_name = os.path.splitext(image_name)[0]

    # signal which file is processed
    feedback_queue.put([script_name, 'Processing: ' + spectral_array.filename])

    # copy unaltered pseudo rgb image for plotting results/debug information on it later
    img_plant_labelled = np.copy(spectral_array.pseudo_rgb)
    img_roi_labelled = np.copy(spectral_array.pseudo_rgb)

    # process ROI items forwarded from the UI
    rois = process_rois(roi_items, img_roi_labelled)

    # identify objects in the ROIs
    labeled_objects, n_obj = pcv.create_labels(mask=mask, rois=rois, roi_type="partial")

    # analyzing objects
    if analyze_index:
        index_functions = rayn_utils.get_index_functions()
        index_array = index_functions[selected_index][1](spectral_array, 10)
        pcv.analyze.spectral_index(index_img=index_array,
                                   labeled_mask=labeled_objects,
                                   n_labels=n_obj,
                                   label="plant")

    if analyze_shape:
        img_plant_labelled = pcv.analyze.size(img=img_plant_labelled,
                                              labeled_mask=labeled_objects,
                                              n_labels=n_obj,
                                              label="plant")

    # return preview image
    image_file_name = os.path.normpath(out_folder + "/ProcessedImages/" + image_name + ".png")
    path, file_name = os.path.split(image_file_name)

    if not os.path.exists(path):
        os.makedirs(path)
        print("created folder " + path)

    print("Writing image to " + image_file_name)

    pcv.print_image(img=img_plant_labelled, filename=image_file_name)

    # Use feedbackQueue.put to send feedback to the main application
    # feedbackQueue.put([name, 'Processing images...'])
    print("writing info to queue")
    feedback_queue.put([script_name, 'preview', image_file_name])

    print("Workflow done")

    # ANALYSIS WORKFLOW END
    # TODO: change how results are saved when the new PlantCV Version is published

    # Processing results
    results = pcv.outputs.observations
    # TODO: this is currently very limited and inflexible. Needs to change!
    results_dict = {}
    results_list = []

    index_key = "index_" + selected_index

    if plot_selection == "plot_index" and analyze_index:
        selected_key = "mean_" + index_key
    else:
        selected_key = plot_selection

    for i in range(1, n_obj + 1):

        if f"plant_{i}" in results:
            roi_results = results[f"plant_{i}"]

            if analyze_shape and not analyze_index:
                results_list.append({"roi": i,
                                     "area": roi_results["area"]["value"],
                                     "width": roi_results["width"]["value"],
                                     "height": roi_results["height"]["value"],
                                     "perimeter": roi_results["perimeter"]["value"],
                                     "index": None,
                                     "mean": None,
                                     "median": None,
                                     "std": None,
                                     "plot_value": roi_results[selected_key]["value"]})

            if analyze_index and not analyze_shape:
                results_list.append({"roi": i,
                                     "area": None,
                                     "width": None,
                                     "height": None,
                                     "perimeter": None,
                                     "index": selected_index,
                                     "mean": roi_results["mean_" + index_key]["value"],
                                     "median": roi_results["med_" + index_key]["value"],
                                     "std": roi_results["std_" + index_key]["value"],
                                     "plot_value": roi_results[selected_key]["value"]})

            if analyze_shape and analyze_index:
                results_list.append({"roi": i,
                                     "area": roi_results["area"]["value"],
                                     "width": roi_results["width"]["value"],
                                     "height": roi_results["height"]["value"],
                                     "perimeter": roi_results["perimeter"]["value"],
                                     "index": selected_index,
                                     "mean": roi_results["mean_" + index_key]["value"],
                                     "median": roi_results["med_" + index_key]["value"],
                                     "std": roi_results["std_" + index_key]["value"],
                                     "plot_value": roi_results[selected_key]["value"]})

    results_dict["rois"] = results_list

    # signal results
    signal_dict = {"imageFileName": image_file_name, "dict": results_dict}
    feedback_queue.put([script_name, 'results', signal_dict])


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


def process_rois(roi_items, rgb_image):  # get the rois from individual coordinates
    # creating empty ROI object
    rois = pcv.Objects(contours=[], hierarchy=[])

    for roi_type, roi_x, roi_y, roi_width, roi_height in roi_items:
        print("RoiItem:", roi_type, roi_x, roi_y, roi_width, roi_height)

        if roi_type == "Circle":
            roi_radius = int(roi_width / 2)
            # create a single circular roi
            roi = pcv.roi.circle(x=roi_x, y=roi_y, r=roi_radius, img=rgb_image)
        elif roi_type == "Rectangle":
            # create a single rectangle roi
            print("calculated x/y", roi_x - roi_width / 2, roi_y - roi_height / 2)
            roi = pcv.roi.rectangle(x=roi_x - roi_width / 2, y=roi_y - roi_height / 2,
                                    h=roi_height, w=roi_width, img=rgb_image)

        else:
            warnings.warn("Roi type is neither circle or rectangle")
            break

        # append the roi contour and hierarchy to the object collecting all the rois
        rois.append(roi.contours, roi.hierarchy)

    return rois


def prepare_spectral_data(settings):
    # file and folder
    img_file = settings["inputImage"]
    # undistort and normalize
    image_options = settings["experimentSettings"]["imageOptions"]

    lens_angle = image_options["lensAngle"]
    dark_normalize = image_options["normalize"]

    # check if a .hdr file name was provided and set img_file to the binary location
    if os.path.splitext(img_file)[1] == ".hdr":
        img_file = os.path.splitext(img_file)[0]

    else:
        warnings.warn("No header file provided. Processing not possible.")
        return

    # begin masking workflow
    spectral_data = pcv.readimage(filename=img_file, mode='envi')
    spectral_data.array_data = spectral_data.array_data.astype("float32")  # required for further calculations
    if spectral_data.d_type == np.uint8:  # only convert if data seems to be uint8
        spectral_data.array_data = spectral_data.array_data / 255  # convert 0-255 (orig.) to 0-1 range

    # normalize the image cube
    if dark_normalize:
        spectral_data.array_data = rayn_utils.dark_normalize_array_data(spectral_data)

    # undistort the image cube
    if lens_angle != 0:  # only undistort if angle is selected
        cam_calibration_file = f"calibration_data/{lens_angle}_calibration_data.yml"  # select the data set
        mtx, dist = rayn_utils.load_coefficients(cam_calibration_file)  # depending on the lens angle
        spectral_data.array_data = rayn_utils.undistort_data_cube(spectral_data.array_data, mtx, dist)
        spectral_data.pseudo_rgb = rayn_utils.undistort_data_cube(spectral_data.pseudo_rgb, mtx, dist)

    return spectral_data


def create_mask_preview(mask, settings, create_preview=True):
    if create_preview:
        out_image = settings["outputImage"]
        image_file_name = os.path.normpath(out_image)
        print("Writing image to " + image_file_name)
        pcv.print_image(img=mask, filename=image_file_name)

