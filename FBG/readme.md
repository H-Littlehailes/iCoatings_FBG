# FBG

Container for the files relating to the strain measurements

output from the Interrogator coupled to the FBG fibres. 
The output from the interrogator is a text file containing the date and time of measurement, the hardware information from the interrogator, the measurement parameters (integration time, sampling rate, noise threshold, sensitivity setting) and the wavelengths of the detected gratings.

Below this is the raw data from the fibres


The python scripts in this folder are intended to import the relevant data files and perform calculations to convert the detected wavelengths to strain or temperature values, depending on the fibre.

Below is a brief description of the function of each file:

## Contents 
- [DTG_Data_Correction_20240326.py](DTG_Data_Correction_20240326.py):
- [DTG_Data_Correction_draft2_20240327.py](DTG_Data_Correction_draft2_20240327.py):
- [DTG_Data_Sorting_Script_20240417.py](DTG_Data_Sorting_Script_20240417.py):
- [DTG_Fibre_Analysis](DTG_Fibre_Analysis):
- [New_Temp_Strain_Compensation.py](New_Temp_Strain_Compensation.py): File calculates the temperature-compensated strain measurement from combination of strain and temperature probe co-measuring in the experiment. File reads in data from the temperature probe and calculates the temperature measured in the experiment, this value is then fed in to the calculation of the strain measured by the strain gauge accounting for the thermally induced strain, to isolate the values of the mechanical strain. Output is a 3x1 plot (i) comparing the temperature-compensated and uncompensated strain, (ii) the temperature measured during the experiment (iii) the difference between strain values in (i) to determine functioning of the calculation - (iii) should match form of (ii)
- [Peak_finder_Stripe5.py](Peak_finder_Stripe5.py):
- [Plate1_Temp_Comp.py](Plate1_Temp_Comp.py):
- [Plate2_Temp_Comp.py](Plate2_Temp_Comp.py):
- [StrainProbeAnalysis.py](StrainProbeAnalysis.py):
- [Strain_Comp_From_TextFile_data.py](Strain_Comp_From_TextFile_data.py):
- [Strain_Comp_From_TextFile_data2.py](Strain_Comp_From_TextFile_data2.py):
- [Strain_Temp_Plotter.py](Strain_Temp_Plotter.py):
- [Strain_Transfer_Error_Wang.py](Strain_Transfer_Error_Wang.py):
- [Strain_Transfer_Error_Wang_2016.py](Strain_Transfer_Error_Wang_2016.py):
- [Stripe5_Midpoint_Fitting.py](Stripe5_Midpoint_Fitting.py):
- [TemperatureChainAnalysis.py](TemperatureChainAnalysis.py): File reads in the data from the temperature chain. The temperature chain fibres contain 3 gratings at different wavelengths, with 3 data streams, and are indexed as '%data_channel_#% - %grating_Number_Low_to_High%'. This script will calculate the temperature variation experienced by each grating and generate a plot.
- [TemperatureComparison_20230905.py](TemperatureComparison_20230905.py): File reads in data from a temperature chain fibre and a temperature probe, for a total of 4 data streams. This script is used to compare the congruency in temperature measurement between fibres. It should be expected that the temperature probe will be more sensitive to temperature variation due to a thinner metal jacket, and the temperature chain has been seen to measure a higher temperature than the temperature probe in the same conditions.
- [TemperatureProbeAnalysis.py](TemperatureProbeAnalysis.py): File reads in the data from the temperature probe. The temperature probe contains 1 grating, with one data stream. This script will calculate the temperature from this grating.
- [Temperature_Compensated_Strain_20240109.py](Temperature_Compensated_Strain_20240109.py):


StrainProbeAnalysis.py  -  File reads in the data from the strain gauge and calculates the strain values directly. The strain gauges are single grating with only one data stream. This file uses single input and does not account for thermally induced strain.

TemperatureChainAnalysis.py  -  File reads in the data from the temperature chain. The temperature chain fibres contain 3 gratings at different wavelengths, with 3 data streams, and are indexed as '%data_channel_#% - %grating_Number_Low_to_High%'. This script will calculate the temperature variation experienced by each grating and generate a plot.

TemperatureProbeAnalysis.py  -  File reads in the data from the temperature probe. The temperature probe contains 1 grating, with one data stream. This script will calculate the temperature from this grating.

TemperatureComparison_20230905.py  -  File reads in data from a temperature chain fibre and a temperature probe, for a total of 4 data streams. This script is used to compare the congruency in temperature measurement between fibres. It should be expected that the temperature probe will be more sensitive to temperature variation due to a thinner metal jacket, and the temperature chain has been seen to measure a higher temperature than the temperature probe in the same conditions.

New_Temp_Strain_Compensation.py  -  File calculates the temperature-compensated strain measurement from combination of strain and temperature probe co-measuring in the experiment. File reads in data from the temperature probe and calculates the temperature measured in the experiment, this value is then fed in to the calculation of the strain measured by the strain gauge accounting for the thermally induced strain, to isolate the values of the mechanical strain. Output is a 3x1 plot (i) comparing the temperature-compensated and uncompensated strain, (ii) the temperature measured during the experiment (iii) the difference between strain values in (i) to determine functioning of the calculation - (iii) should match form of (ii)
