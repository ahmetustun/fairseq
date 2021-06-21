#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse, os
import xml.etree.ElementTree as ET

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="the path of the dataset folder")
    args = parser.parse_args()

    os.mkdir(os.path.join(args.dir, 'preprocess'))

    files = os.listdir(args.dir)

    for file in files:
        if file.startswith('train'):
            with open(os.path.join(args.dir,'preprocess', file), 'w') as write:
                with open(os.path.join(args.dir, file)) as read:
                    for line in read:
                        if not line.lstrip().startswith('<'):
                            write.write(line.strip()+'\n')

        if file.endswith('.xml'):
            with open(os.path.join(args.dir, 'preprocess', file[:-4]), 'w') as write:
                xml = ET.parse(os.path.join(args.dir, file))
                root = xml.getroot()
                for doc in root.iter('doc'):
                    for seg in doc.iter('seg'):
                        write.write(seg.text.strip()+'\n')

if __name__ == "__main__":
    main()

