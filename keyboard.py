# coding: utf-8
import serial
import time
import os
from threading import Thread

class Keyboard(Thread):
    serial_device = '/dev/ttyACM0' # cu.usbmodemFA131 on macOS

    def __init__(self):
        Thread.__init__(self)
        self.ser = None
        self.is_running = True

    def run(self):
        while self.is_running:
            self.openAndListen()
            if self.is_running:
                print("Connecting again in 1s...")
                time.sleep(1)
            else:
                print("Serial link turning off.")

    def openAndListen(self):
        if not self.isAvailable():
            print("Device not connected")
            return

        print("Connecting to serial...")
        self.connect()
        time.sleep(1)
        print("Connected.")

        # Write data
        #self.ser.write(str.encode("COUCOU"))
        #self.ser.write(str.encode("\n"))

        #while self.is_running:
        #    try:
        #        received = self.ser.readline()
        #        if len(received)> 0:
        #            print ("=> %s" % received.replace('\n', ''))
        #    except:
        #        print ("Serial link closed")
        #        break
        #self.ser.close()

    def isAvailable(self):
        return os.access(self.serial_device, os.R_OK)

    def connect(self):
        self.ser = serial.Serial(self.serial_device, 9600, timeout=10)

    def stopOnJoin(self):
        self.is_running = False
        self.ser.flush()
        self.ser.close()

    def send(self, data):
        if not self.isAvailable():
            print("    [KEYBOARD] Trying to send to disconnected device.")
            self.ser = None
            while self.ser == None:
                print("    [KEYBOARD] Reconnecting...")
                try:
                    self.connect()
                except:
                    print("    [KEYBOARD] Failed to reconnect for now")
                time.sleep(1)
            print("    [KEYBOARD] Reconnected.")
            time.sleep(3)
        #print (data)
        self.ser.write(str.encode(data))
        #self.ser.write(str.encode("\n"))
