#include <HID.h>
#include <KeyboardAzertyFr.h>

String inputString = ""; // a String to hold incoming data
bool stringComplete = false;  // whether the string is complete

// Runs once when you press reset or power the board
void setup() {
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(9600);

  // reserve 200 bytes for the inputString:
  inputString.reserve(200);

  // Open the Keyboard link
  KeyboardAzertyFr.begin();
}

// Runs over and over again forever
void loop() {
  
  // When a new line arrives
  if (stringComplete) {
    digitalWrite(LED_BUILTIN, HIGH); // turn the LED on

    // Send it back via serial
    Serial.print("Received : ");
    Serial.println(inputString);
    //delay(1000);
    // Send it via Keyboard
    KeyboardAzertyFr.print(inputString);
    KeyboardAzertyFr.print('\n');
    Serial.println("Sent via keyboard.");
    
    // Clear the string
    inputString = "";
    stringComplete = false;
    
    digitalWrite(LED_BUILTIN, LOW); // turn the LED off
  }

  // Listen for incoming string
  while (Serial.available() > 0) {
     inputString = Serial.readStringUntil('\n');
     stringComplete = true;
  }
}
