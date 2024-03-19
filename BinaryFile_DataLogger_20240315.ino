#include <Adafruit_MAX31856.h>
#include <SD.h>
#include <SPI.h>
#include <Wire.h>
#include <RTClib.h>

#define CS_PIN 17
RTC_DS1307 rtc;

const int chipSelect = 10; // Pin for SD card chip select
File dataFile;
unsigned long lastFlushTime = 0;
const unsigned long flushInterval = 1000; // Flush data every 1 seconds

// Define the thermocouple sensors
Adafruit_MAX31856 maxthermo = Adafruit_MAX31856(CS_PIN, 7, 8, 9);  //9    //6
Adafruit_MAX31856 maxthermo2 = Adafruit_MAX31856(6, 7, 8, 9);  //8   //5
Adafruit_MAX31856 maxthermo3 = Adafruit_MAX31856(5, 7, 8, 9);  //7   //4
Adafruit_MAX31856 maxthermo4 = Adafruit_MAX31856(4, 7, 8, 9);  //5   //2
Adafruit_MAX31856 maxthermo5 = Adafruit_MAX31856(2, 7, 8, 9);  //

const char SEP = ",";
const char* SEP_PTR = &SEP;

void setup() {
  Serial.begin(115200);

  // Initialize RTC
  if (!rtc.begin()) {
    Serial.println("Couldn't find RTC");
    while (1);
  }
  rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));  
  // Initialize SD card
  if (!SD.begin(chipSelect)) {
    Serial.println("SD card initialization failed!");
    return;
  }
  // Generate filename based on current date
  DateTime now = rtc.now();
  char filename[13];
  snprintf(filename, sizeof(filename), "%02d%02d%02d%02d.bin", now.month(), now.day(), now.hour(), now.minute());
  Serial.println(filename);
  // Open the data file
  dataFile = SD.open(filename, O_WRITE | O_CREAT);
  if (!dataFile) {
    Serial.println("Error opening data file!");
    return;
  }
  Serial.println("MAX31856 thermocouple test");
  // Initialize thermocouples
  maxthermo.begin();
  maxthermo.setThermocoupleType(MAX31856_TCTYPE_N);
  maxthermo2.begin();
  maxthermo2.setThermocoupleType(MAX31856_TCTYPE_N);
  maxthermo3.begin();
  maxthermo3.setThermocoupleType(MAX31856_TCTYPE_N);
  maxthermo4.begin();
  maxthermo4.setThermocoupleType(MAX31856_TCTYPE_N);
  maxthermo5.begin();
  maxthermo5.setThermocoupleType(MAX31856_TCTYPE_N);
}
String DateStamp;
//String Milliseconds;
void loop() {
  //unsigned long startMillis = millis(); // Capture start milliseconds
  //while (millis() - startMillis < 1000) {} // Wait for the nearest second
  DateTime now = rtc.now();//.unixtime();
  //now.unixtime()
  //int milliseconds = millis() % 1000; // Get milliseconds
  //float temp_1 = thermocouple_1.readThermocoupleTemperature();
  //float temp_2 = thermocouple_2.readThermocoupleTemperature();
  //float temp_3 = thermocouple_3.readThermocoupleTemperature();
  //float temp_4 = thermocouple_4.readThermocoupleTemperature();
  //float temp_5 = thermocouple_5.readThermocoupleTemperature();

  float c1 = maxthermo.readCJTemperature();
  float h1 = maxthermo.readThermocoupleTemperature();
  float c2 = maxthermo2.readCJTemperature();
  float h2 = maxthermo2.readThermocoupleTemperature();
  float c3 = maxthermo3.readCJTemperature();
  float h3 = maxthermo3.readThermocoupleTemperature();
  float c4 = maxthermo4.readCJTemperature();
  float h4 = maxthermo4.readThermocoupleTemperature();
  float c5 = maxthermo5.readCJTemperature();
  float h5 = maxthermo5.readThermocoupleTemperature();

  // Write data to the SD card
  DateStamp = String(now.unixtime());
  //String(now.year(), DEC) + "/" + 
  //String(now.month(), DEC) + "/" + 
  //String(now.day(), DEC) + " " + 
  //String(now.hour(), DEC) + ":" + 
  //String(now.minute(), DEC) + ":" + 
  //String(now.second(), DEC);
  
  //if (milliseconds < 10) {
  //  Serial.print("00");
  //} else if (milliseconds < 100) {
  //  Serial.print("0");
  // }
  //Milliseconds = String(milliseconds, DEC);

  dataFile.write(String(DateStamp).c_str());
  dataFile.write(SEP);
  //dataFile.write(String(Milliseconds).c_str());
  //dataFile.write(".");
  dataFile.write(String(c1,4).c_str());
  dataFile.write(SEP);
  dataFile.write(String(h1,4).c_str());
  dataFile.write(SEP);
  dataFile.write(String(c2,4).c_str());
  dataFile.write(SEP);
  dataFile.write(String(h2,4).c_str());
  dataFile.write(SEP);
  dataFile.write(String(c3,4).c_str());
  dataFile.write(SEP);
  dataFile.write(String(h3,4).c_str());
  dataFile.write(SEP);
  dataFile.write(String(c4,4).c_str());
  dataFile.write(SEP);
  dataFile.write(String(h4,4).c_str());
  dataFile.write(SEP);
  dataFile.write(String(c5,4).c_str());
  dataFile.write(SEP);
  dataFile.write(String(h5,4).c_str());
  dataFile.write("\n");
  /*
  dataFile.write((uint8_t*)&now, sizeof(now));
  dataFile.write((uint8_t*)&c1, sizeof(c1));
  dataFile.write((uint8_t*)&h1, sizeof(h1));
  dataFile.write((uint8_t*)&c2, sizeof(c2));
  dataFile.write((uint8_t*)&h2, sizeof(h2));
  dataFile.write((uint8_t*)&c3, sizeof(c3));
  dataFile.write((uint8_t*)&h3, sizeof(h3));
  dataFile.write((uint8_t*)&c4, sizeof(c4));
  dataFile.write((uint8_t*)&h4, sizeof(h4));
  dataFile.write((uint8_t*)&c5, sizeof(c5));
  dataFile.write((uint8_t*)&c5, sizeof(h5));
  */
  // Periodically flush data to the SD card
  unsigned long currentMillis = millis();
  if (currentMillis - lastFlushTime >= flushInterval) {
    dataFile.flush();
    lastFlushTime = currentMillis;
  }
  //// Print data to serial monitor
  Serial.print(now.timestamp());
  Serial.print(",");
  Serial.println(c1);
  //Serial.print(",");
  //Serial.print(h1);
  //Serial.print(",");
  //Serial.print(c2);
  //Serial.print(",");
  //Serial.print(h2);
  //Serial.print(",");
 // Serial.print(c3);
  //Serial.print(",");
  //Serial.print(h3);
 // Serial.print(",");
 // Serial.print(c4);
 // Serial.print(",");
 // Serial.print(h4);
 // Serial.print(",");
 // Serial.print(c5);
 // Serial.print(",");
 // Serial.print(h5);

  delay(1); // Adjust delay as needed for sampling rate
}

void closeFile() {
  dataFile.close();
}
