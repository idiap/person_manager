# coding=utf-8

import pprint
from misc import load_mot_annotations

file_name = "mot.csv"

annotations = load_mot_annotations(file_name)

pprint.pprint(annotations)
