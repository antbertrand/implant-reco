#!/usr/bin/env python
# encoding: utf-8
"""
.py
Created by Pierre-Julien Grizel et al.
Copyright (c) 2016 NumeriCube. All rights reserved.
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

import json
import os
import argparse
import functools
import sys

import glob

import cv2
import xml.etree.ElementTree as ET



def json_to_csv(path, save_path):
    """Convert JSON to CSV
    """
    letters_classe_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '/', 'a', 'b', 'c', 'd', 'e',
    'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    i = 0
    #f = open('train.txt', 'w+')
    for json_file in glob.glob(path + "/*.json"):
        with open(json_file) as f:
            json_data = json.load(f)

        f = open(str(save_path), 'a')
        for obj in json_data['objects']:

            filename = ("%s.png"% json_file[:-5]).replace('ann', 'img')
            xmin = int(obj['points']['exterior'][0][0])
            xmax = int(obj['points']['exterior'][1][0])
            ymin = int(obj['points']['exterior'][0][1])
            ymax = int(obj['points']['exterior'][1][1])

            if obj['classTitle'].startswith('pastille') or obj['classTitle'].startswith('texte') :
                classe = int(0)
            else:
                classe = letters_classe_list.index(obj['classTitle'])

            i += 1
            f.write(str(filename) + " " + str(xmin) + "," + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + ',' + str(classe) + '\n')
    f.close()

    print ("Conversion de " + str(i) + " done")


def parse_args(args):
    """Parse input args"""

    def csv_list(string):
        return string.split(",")

    parser = argparse.ArgumentParser(
        description="""Convert JSON directory into text labels.
    Takes a whole directory as in input, randomly split into 3 datasets (train, val and x-val).
    On Paperspace, with a reasonable dataset, it's clever to put these into /artifacts
        so you can inspect later what datasets you were training with.
    """
    )
    parser.add_argument("images_dir", help="Images directory")
    parser.add_argument("json_dir", help="JSON annotations directory")
    parser.add_argument("save_dir", help="Texte annotations directory")
    #parser.add_argument("output_dir", help="Generated CSV+cropped images path")

    return parser.parse_args(args)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Let's go!
    json_to_csv(args.json_dir, args.save_dir)


main()
