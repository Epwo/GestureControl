#include <Arduino.h>
#include <ArduinoJson.h>
#include <SPIFFS.h>

// Pin configuration
#define IR_RECEIVE_PIN 2  // IR receiver connected to GPIO pin 2

// Define file path for JSON storage
#define JSON_FILE_PATH "/remote_codes.json"

// Global variables
String currentCommand = "";
bool awaitingLabel = false;
uint32_t lastIRCode = 0;
volatile unsigned long irData = 0;
volatile int bitCount = 0;
volatile boolean irReceiving = false;
volatile unsigned long lastTime = 0;

// Function prototypes
void setupIRReceiver();
void setupSPIFFS();
bool saveIRCodeToJSON(uint32_t code, String label);
void listAllCodes();
void processSerialCommand();
void displayHelp();
void IRAM_ATTR handleIRInterrupt();

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  
  Serial.println("IR Remote Code Recorder");
  Serial.println("------------------------");
  
  // Initialize SPIFFS
  setupSPIFFS();
  
  // Initialize IR receiver
  setupIRReceiver();
  
  displayHelp();
}

void loop() {
  // Process IR data if a complete code was received
  if (bitCount >= 32) {
    lastIRCode = irData;
    Serial.print("Received IR Code: 0x");
    Serial.println(lastIRCode, HEX);
    
    if (!awaitingLabel) {
      Serial.println("Type 'save [LABEL]' to save this code with a label");
    }
    
    // Reset for next code
    bitCount = 0;
    irData = 0;
  }
  
  // Check if serial input available
  if (Serial.available()) {
    processSerialCommand();
  }
}

void setupIRReceiver() {
  pinMode(IR_RECEIVE_PIN, INPUT);
  attachInterrupt(digitalPinToInterrupt(IR_RECEIVE_PIN), handleIRInterrupt, CHANGE);
  Serial.println("IR Receiver enabled");
}

// Interrupt handler for IR signal
void IRAM_ATTR handleIRInterrupt() {
  unsigned long currentTime = micros();
  unsigned long duration = currentTime - lastTime;
  lastTime = currentTime;
  
  // Basic NEC protocol decoding
  // Typical NEC protocol: 9ms leading pulse, 4.5ms space, then data bits
  
  if (duration > 8000 && duration < 10000) {
    // Start of new transmission (NEC leading pulse ~9ms)
    irReceiving = true;
    bitCount = 0;
    irData = 0;
  } 
  else if (irReceiving) {
    if (duration > 1000 && duration < 2000) {
      // Bit 1 (~1.6ms space)
      irData = (irData << 1) | 1;
      bitCount++;
    } 
    else if (duration > 400 && duration < 700) {
      // Bit 0 (~0.56ms space)
      irData = (irData << 1);
      bitCount++;
    }
    
    // If we've received all bits, stop receiving
    if (bitCount >= 32) {
      irReceiving = false;
    }
  }
}

void setupSPIFFS() {
  if (!SPIFFS.begin(true)) {
    Serial.println("An error occurred while mounting SPIFFS");
    return;
  }
  Serial.println("SPIFFS mounted successfully");
  
  // Create JSON file if it doesn't exist
  if (!SPIFFS.exists(JSON_FILE_PATH)) {
    File file = SPIFFS.open(JSON_FILE_PATH, FILE_WRITE);
    if (!file) {
      Serial.println("Failed to create file");
      return;
    }
    // Initialize with empty JSON object
    file.println("{}");
    file.close();
    Serial.println("Created empty JSON file");
  }
}

bool saveIRCodeToJSON(uint32_t code, String label) {
  // Read existing JSON file
  File file = SPIFFS.open(JSON_FILE_PATH, FILE_READ);
  if (!file) {
    Serial.println("Failed to open file for reading");
    return false;
  }
  
  // Parse existing JSON
  DynamicJsonDocument doc(4096);  // Adjust size as needed
  DeserializationError error = deserializeJson(doc, file);
  file.close();
  
  if (error) {
    Serial.print("deserializeJson() failed: ");
    Serial.println(error.c_str());
    return false;
  }
  
  // Add or update the code
  String codeStr = "0x" + String(code, HEX);
  doc[codeStr] = label;
  
  // Write back to file
  file = SPIFFS.open(JSON_FILE_PATH, FILE_WRITE);
  if (!file) {
    Serial.println("Failed to open file for writing");
    return false;
  }
  
  if (serializeJson(doc, file) == 0) {
    Serial.println("Failed to write to file");
    file.close();
    return false;
  }
  
  file.close();
  return true;
}

void listAllCodes() {
  File file = SPIFFS.open(JSON_FILE_PATH, FILE_READ);
  if (!file) {
    Serial.println("Failed to open file for reading");
    return;
  }
  
  DynamicJsonDocument doc(4096);
  DeserializationError error = deserializeJson(doc, file);
  file.close();
  
  if (error) {
    Serial.print("deserializeJson() failed: ");
    Serial.println(error.c_str());
    return;
  }
  
  Serial.println("\nStored IR Codes:");
  Serial.println("----------------");
  
  JsonObject obj = doc.as<JsonObject>();
  for (JsonPair kv : obj) {
    Serial.print(kv.key().c_str());
    Serial.print(" : ");
    Serial.println(kv.value().as<String>());
  }
  Serial.println();
}

void processSerialCommand() {
  char c = Serial.read();
  
  if (c == '\n') {
    // Process the completed command
    currentCommand.trim();
    
    if (currentCommand.startsWith("save ")) {
      String label = currentCommand.substring(5);
      if (label.length() > 0 && lastIRCode != 0) {
        if (saveIRCodeToJSON(lastIRCode, label)) {
          Serial.print("Saved code 0x");
          Serial.print(lastIRCode, HEX);
          Serial.print(" with label '");
          Serial.print(label);
          Serial.println("'");
        } else {
          Serial.println("Failed to save code");
        }
      } else {
        Serial.println("Invalid label or no IR code received yet");
      }
    } 
    else if (currentCommand == "list") {
      listAllCodes();
    } 
    else if (currentCommand == "clear") {
      File file = SPIFFS.open(JSON_FILE_PATH, FILE_WRITE);
      if (file) {
        file.println("{}");
        file.close();
        Serial.println("All stored codes cleared");
      } else {
        Serial.println("Failed to clear codes");
      }
    } 
    else if (currentCommand == "help") {
      displayHelp();
    } 
    else if (currentCommand.length() > 0) {
      Serial.print("Unknown command: ");
      Serial.println(currentCommand);
      Serial.println("Type 'help' for available commands");
    }
    
    currentCommand = "";
  } else {
    // Add character to current command
    currentCommand += c;
  }
}

void displayHelp() {
  Serial.println("\nAvailable Commands:");
  Serial.println("------------------");
  Serial.println("save [LABEL] - Save the last received IR code with the given label");
  Serial.println("list         - List all stored IR codes and their labels");
  Serial.println("clear        - Delete all stored codes");
  Serial.println("help         - Show this help message");
  Serial.println();
}
