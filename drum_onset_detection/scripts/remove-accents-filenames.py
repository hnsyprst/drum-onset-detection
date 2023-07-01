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

def remove_others(s):
    nd_charmap = {
        u'\N{Latin capital letter AE}': 'AE',
        u'\N{Latin small letter ae}': 'ae',
        u'\N{Latin capital letter Eth}': 'D',
        u'\N{Latin small letter eth}': 'd',
        u'\N{Latin capital letter O with stroke}': 'O',
        u'\N{Latin small letter o with stroke}': 'o', 
        u'\N{Latin capital letter Thorn}': 'Th',
        u'\N{Latin small letter thorn}': 'th',
        u'\N{Latin small letter sharp s}': 's',
        u'\N{Latin capital letter D with stroke}': 'D',
        u'\N{Latin small letter d with stroke}': 'd',
        u'\N{Latin capital letter H with stroke}': 'H',
        u'\N{Latin small letter h with stroke}': 'h',
        u'\N{Latin small letter dotless i}': 'i',
        u'\N{Latin small letter kra}': 'k',
        u'\N{Latin capital letter L with stroke}': 'L',
        u'\N{Latin small letter l with stroke}': 'l',
        u'\N{Latin capital letter Eng}': 'N',
        u'\N{Latin small letter eng}': 'n',
        u'\N{Latin capital ligature OE}': 'Oe',
        u'\N{Latin small ligature oe}': 'oe',
        u'\N{Latin capital letter T with stroke}': 'T',
        u'\N{Latin small letter t with stroke}': 't'}
    
    "Removes diacritics from the string"
    b=[]
    for ch in s:
        if  unicodedata.category(ch)!= 'Mn':
            if ch in nd_charmap:
                b.append(nd_charmap[ch])
            elif ord(ch)<128:
                b.append(ch)
            else:
                b.append(' ')
    return ''.join(b)

def renameit(directory, name):
    new_name = remove_accents(name)
    new_name = remove_others(new_name)
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