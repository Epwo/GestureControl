#include <Arduino.h>
#include <IRremote.h>
#include <ArduinoJson.h>
#include <SPIFFS.h>

// Pin configuration
#define IR_RECEIVE_PIN 2  // IR receiver connected to GPIO pin 15

// Define file path for JSON storage
#define JSON_FILE_PATH "/remote_codes.json"

// Global variables
IRrecv irReceiver(IR_RECEIVE_PIN);
decode_results irResults;
String currentCommand = "";
bool awaitingLabel = false;
uint32_t lastIRCode = 0;

// Function prototypes
void setupIRReceiver();
void setupSPIFFS();
bool saveIRCodeToJSON(uint32_t code, String label);
void listAllCodes();
void processSerialCommand();
void displayHelp();

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
  // Check if IR signal received
  if (irReceiver.decode(&irResults)) {
    if (irResults.decode_type != UNKNOWN) {
      lastIRCode = irResults.value;
      Serial.print("Received IR Code: 0x");
      Serial.println(lastIRCode, HEX);
      
      if (!awaitingLabel) {
        Serial.println("Type 'save [LABEL]' to save this code with a label");
      }
    }
    irReceiver.resume(); // Receive the next value
  }
  
  // Check if serial input available
  if (Serial.available()) {
    processSerialCommand();
  }
}

void setupIRReceiver() {
  irReceiver.enableIRIn();
  Serial.println("IR Receiver enabled");
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
