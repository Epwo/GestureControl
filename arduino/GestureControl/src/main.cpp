#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <IRremote.h>  // Include IRremote library

// Pin configuration
#define IR_SEND_PIN 14 // IR LED connected to GPIO pin 14

// WiFi credentials
const char* ssid = "Banane";
const char* password = "CoolSwag";
// Web server on port 80
WebServer server(80);

// Create IR sender object
IRsend irsend(IR_SEND_PIN);

// Function prototypes
void setupWiFi();
void setupEndpoints();
void handleRoot();
void handleSendIR();

void setup() {
  Serial.begin(9600);
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  
  Serial.println("IR Code Sender API");
  Serial.println("------------------");
  
  // Start IRremote
  IrSender.begin(IR_SEND_PIN);
  
  // Connect to WiFi
  setupWiFi();
  
  // Setup web server endpoints
  setupEndpoints();
  
  // Start server
  server.begin();
  Serial.println("HTTP server started");
}

void loop() {
  // Handle client requests
  server.handleClient();
}

void setupWiFi() {
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.mode(WIFI_STA);
  
  // Scan for available networks
  Serial.println("Scanning for WiFi networks...");
  int networksFound = WiFi.scanNetworks();
  
  if (networksFound == 0) {
    Serial.println("No networks found");
  } else {
    Serial.print(networksFound);
    Serial.println(" networks found:");
    for (int i = 0; i < networksFound; ++i) {
      // Print SSID and RSSI for each network found
      Serial.print(i + 1);
      Serial.print(": ");
      Serial.print(WiFi.SSID(i));
      Serial.print(" (");
      Serial.print(WiFi.RSSI(i));
      Serial.print(" dBm)");
      Serial.println((WiFi.encryptionType(i) == WIFI_AUTH_OPEN) ? " [Open]" : " [Secured]");
      delay(10);
    }
  }
  Serial.println("");
  
  // Check if configured SSID is available
  bool ssidFound = false;
  for (int i = 0; i < networksFound; i++) {
    if (WiFi.SSID(i) == String(ssid)) {
      ssidFound = true;
      break;
    }
  }
  
  if (ssidFound) {
    Serial.print("Configured network '");
    Serial.print(ssid);
    Serial.println("' found! Connecting...");
  } else {
    Serial.print("Warning: Configured network '");
    Serial.print(ssid);
    Serial.println("' not found in scan results!");
  }
  WiFi.begin(ssid, password);
  
  // Add timeout for WiFi connection
  int timeout = 30; // 30 * 500ms = 15 seconds timeout
  while (WiFi.status() != WL_CONNECTED && timeout > 0) {
    delay(500);
    Serial.print(".");
    timeout--;
  }
  
  Serial.println("");
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("WiFi connected");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("WiFi connection failed!");
    Serial.print("Status code: ");
    Serial.println(WiFi.status());
    // Continue anyway to allow connection troubleshooting
  }
}

void setupEndpoints() {
  server.on("/", HTTP_GET, handleRoot);
  server.on("/send", HTTP_POST, handleSendIR);
  
  // Handle not found
  server.onNotFound([]() {
    server.send(404, "text/plain", "Not found");
  });
}

void handleRoot() {
  Serial.println("Received request on root");
  String html = "<html><head><title>IR Sender API</title></head>";
  html += "<body><h1>IR Sender API</h1>";
  html += "<p>Use POST /send with JSON payload: {\"code\": \"0xYOUR_HEX_CODE\"}</p>";
  html += "<form action='/send' method='post'>";
  html += "IR Code (hex): <input type='text' name='code' placeholder='0x12345678'>";
  html += "<input type='submit' value='Send'>";
  html += "</form></body></html>";
  
  server.send(200, "text/html", html);
}

void handleSendIR() {
  String code;
  uint32_t irCode = 0;
  
  // Check if the request is a form submission or JSON
  if (server.hasArg("code")) {
    // Form submission
    code = server.arg("code");
    irCode = strtoul(code.c_str(), NULL, 16);
  } else {
    // JSON payload
    String json = server.arg("plain");
    DynamicJsonDocument doc(1024);
    DeserializationError error = deserializeJson(doc, json);
    
    if (error) {
      server.send(400, "text/plain", "Invalid JSON");
      return;
    }
    
    if (!doc.containsKey("code")) {
      server.send(400, "text/plain", "Missing 'code' field");
      return;
    }
    
    code = doc["code"].as<String>();
    // Remove '0x' prefix if present
    if (code.startsWith("0x")) {
      code = code.substring(2);
    }
    irCode = strtoul(code.c_str(), NULL, 16);
  }
  
  if (irCode == 0) {
    server.send(400, "text/plain", "Invalid IR code");
    return;
  }
  
  Serial.print("Sending IR code: 0x");
  Serial.println(irCode, HEX);
  
  // Send the IR code using NEC protocol
  irsend.sendNEC(irCode ,32);
  
  // Send response
  server.send(200, "text/plain", "IR code sent: 0x" + String(irCode, HEX));
}
