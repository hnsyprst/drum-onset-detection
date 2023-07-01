#!/usr/bin/env python
# recursively rename international characters in filenames 
# Copyright (C) Vlatko Kosturjak - Kost

import os
import sys
import glob
import unicodedata

startingdir=u'F:\Work2\drum-onset-detection\data\ADTOF-master\dataset'

def remove_accents(s):
    nkfd_form = unicodedata.normalize('NFKD', s)
    return u''.join([c for c in nkfd_form if not unicodedata.combining(c)])

def renameit(directory, name):
    new_name = remove_accents(name)
    if new_name != name:
        try:
            fname=os.path.join(directory, name)
            new_fname=os.path.join(directory, new_name)
            print('R:',fname, '=>', new_fname)
            os.rename(fname, new_fname)
            return new_name
        except Exception as e:
            print(e)
    return None

print('Starting renaming filenames in', startingdir)
for dirname, dirnames, filenames in os.walk(startingdir):
    for subdirname in dirnames:
        fulldir=os.path.join(dirname, subdirname)
        print('D:', fulldir)
        new_name=renameit(dirname, subdirname)
        if new_name is not None:
            dirnames.append(new_name)

    for filename in filenames:
        fullfile=os.path.join(dirname, filename)
        print('F:', fullfile)
        renameit(dirname, filename)