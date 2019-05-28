#!/usr/bin/env python
# encoding: utf-8
"""
eurosilicone-reader.py

Created by Pierre-Julien Grizel et al.
Copyright (c) 2016 NumeriCube. All rights reserved.

The main EUROSILICONE program but without a GUI.

Main features:
* Opens and constantly reads camera
* If something is detected, print a line on the screen
* Writes the image in a quiet place
* Loop forever

"""
from __future__ import unicode_literals

__author__ = ""
__copyright__ = "Copyright 2016, NumeriCube"
__credits__ = ["Pierre-Julien Grizel"]
__license__ = "CLOSED SOURCE"
__version__ = "TBD"
__maintainer__ = "Pierre-Julien Grizel"
__email__ = "pjgrizel@numericube.com"
__status__ = "Production"


class EurosiliconeReader():
    """Singleton class
    """

    def main(self,):
        """Main program loop.
        """
        FORMAT = "%(asctime)s %(message)s"
        logging.basicConfig(
            filename="./activity.log", format=FORMAT, level=logging.DEBUG
        )


        """
        self.cam = camera.Camera()
        # self.past_detection = yolo.YOLO()
        logging.info("Camera detected")
        # self.cam.saveConf()
        # logging.info("Camera configuration saved")
        self.cam.loadConf("acA5472-17um.pfs")
        logging.info("Camera configuration loaded")
        """
        # Start detectors
        self.past_detection = yolo.YOLO()
        self.better_circle =
        self.corr_orientation = orientator.OrientationFixer



        # Until death stikes, we read images continuously.
        print("START READING...")
        while True:
            # Grab image from camera
            fullimg, img = self.cam.grabbingImage()

            # Convert to PIL format
            img_pil = Image.fromarray(fullimg)
            # computeResults = image.Image(fullimg)

            # Pastille detection, save image on-the-fly.
            # The goal here is to perform a FIRST detection, wait for 1s and perform a SECOND detection
            # in order to avoid keeping/getting blurry images.
            is_detected, out_boxes, out_scores, out_classes = self.past_detection.detect_image(
                img_pil
            )
            if not is_detected:
                continue

            print("DON'T MOVE!")
            time.sleep(1)
            fullimg, img = self.cam.grabbingImage()
            is_detected, out_boxes, out_scores, out_classes = self.past_detection.detect_image(
                img_pil
            )

            if not is_detected:
                print("You moved...")
                continue

            # Get detected zone
            self.detect = detection_instance.DetectionInstance(fullimg)
            is_cropped, img_chip = self.detect.get_chip_area(out_boxes)

            # Save chip image
            output_fn = os.path.join(
                ACQUISITIONS_PATH,
                "CHIP-{}.png".format(time.strftime("%Y-%m-%d-%H%M%S")),
            )
            Image.fromarray(img_chip).save(output_fn)
            output_fn = output_fn.replace("CHIP", "FULL")
            Image.fromarray(fullimg).save(output_fn)
            print("Image saved: {}".format(output_fn))
            # img_chip.save(output_fn)

            # Get additional info
            best_angle = self.get_chip_angle(img_chip)
            if best_angle is not None:
                self.get_text_from_azure(img_chip, best_angle)

            # computeResults.saveImage(img_chip)
            print(
                "Image saved. Change/Turn prothesis. Waiting 5s before detecting again."
            )
            time.sleep(5)
            logging.info("Circle detected")

            # self.ui.displayImage(img)
            print("START READING...")
