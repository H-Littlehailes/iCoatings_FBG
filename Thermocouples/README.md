# Thermocouple controller

Files in this folder relate to capturing and processing data from a custom built thermocouple temperature-probe station.
This device measures temperature values from 5 separate type-N high temperature (1200C) thermocouples amplified by
Adafruit_MAX31856 modules, and processed and collated by an Arduino-Uno. The results are saved as a binary file with the
corresponding date-time stamp measured by an RTC_DS1307 rtc.

An outline of the controller devices is shown in the below image
![controller](images/Thermocouple_Controller.png)
