# RAYN Vision System Analytics - Analysis Scripts
Collection of analysis scripts for RAYN Vision System (RVS) Analytics that are maintained by RAYN.

## Description
RVS analysis scripts basically are [PlantCV](https://plantcv.readthedocs.io/en/stable/) workflows. They consist of two
main parts:
1. Object segmentation ("Masking") - separate objects from background
2. Object analysis ("Analysis") - quantify certain object parameters (depending on the experiment)

More information, details and explanations can be found here: [Workflow Development (PlantCV)](
https://plantcv.readthedocs.io/en/stable/analysis_approach/#developing-image-processing-workflows-workflow-development)

## Usage
Each analysis script consists of a folder containing three files:
- SCRIPT_NAME.py - Actual python script, containing the [PlantCV](https://plantcv.readthedocs.io/en/stable/) workflow
- SCRIPT_NAME.config - Configuration file setting the UI elements in the mask UI interface
- README.md - Describes the function and purpose of the respective mask script

Pull this repository or add the script folders to the "Scripts" folder of RVS Analytics.

All available analysis scripts are shown in the respective drop-down menu and can be selected from there. Each analysis 
script has an inbuilt default mask script. However, it is possible to select a different mask scripts from the 
respective mask drop-down menu.

## Support
If you experience any problems or have feedback on the analysis scripts, please [add an issue to this repository](https://github.com/rayngrowingsystems/RVS-A_analysis_scripts/issues)
or contact [RAYN Vision Support](mailto:RAYNVisionSupport@rayngrowingsystems.com).

## Contributing
Whether it's fixing bugs, adding functionality to existing analysis scripts or adding entirely new analysis
scripts, we welcome contributions.

## Create Your Own RVS Analysis Scripts
Instructions on how to create your own mask scrips will be linked here.

## License and Copyright
© 2024 RAYN Growing Systems, All Rights Reserved. Licensed under the Apache License, Version 2.0

Trademark and patent info: [rayngrowingsystems.com/ip](https://rayngrowingsystems.com/ip/) \
Third-party license agreement info: [etcconnect.com/licenses](https://www.etcconnect.com/licenses/) \
Product and specifications subject to change.
