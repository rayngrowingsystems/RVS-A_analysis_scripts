# Base Script
## Description
This is a script providing base functionality. It returns shape parameters and reflectance index information.

## Details
### Default Mask
The default masking script uses a single wavelength band and a threshold to create a binary mask. 
The wavelength band can be selected from a drop-down men and the threshold can be adjusted via a slider.

### Script Settings UI Options
- Index analysis
- Shape analysis
- Set line width of annotations

### Output
**Selected Reflectance Indices**\
See all supported indices in get_index_functions() in rayn_utils.py. The analysis returns mean, median, and 
standard deviation of the respective index. 

Currently, it is not possible to analyse multiple indices with this script.

**Selected Shape Parameters**\
The following shape parameter are currently supported
- area
- perimeter
- width
- height

Although available in PlantCV, it is currently not possible to return more shape parameters.

## Support
If you experience any problems or have feedback on the analysis scripts, please [add an issue to this repository](https://github.com/rayngrowingsystems/RVS-A_analysis_scripts/issues) 
or contact [RAYN Vision Support](mailto:RAYNVisionSupport@rayngrowingsystems.com).


## License and Copyright
© 2024 RAYN Growing Systems, All Rights Reserved. Licensed under the Apache License, Version 2.0

Trademark and patent info: [rayngrowingsystems.com/ip](https://rayngrowingsystems.com/ip/) \
Third-party license agreement info: [etcconnect.com/licenses](https://www.etcconnect.com/licenses/) \
Product and specifications subject to change.