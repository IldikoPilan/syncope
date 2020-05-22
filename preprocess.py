"""
Created on Thu Oct 17 15:42:54 2019
@author: ILDPIL
"""

import pyodbc 
import time
import os
import re
import codecs
from collections import Counter
import numpy as np
import pandas as pd

###########
# Data prep
###########

def fetch_data(sql, connect_info):
    conn = pyodbc.connect(connect_info[0],
                          connect_info[1],
                          connect_info[2],
                          connect_info[3],
                          'charset=utf8')
    data = pd.read_sql(sql, conn)
    return data

def remove_info(text, journal_id, label, doc_type='inkomst'):
    """ Removes information from texts about diagnosis.
    """
    sections = text.split('NEWPAR')
    cleaned_text = ''
    diagnose_detected = False
    for section in sections:
        if section:
            section_header =list(filter(None, section.split(' ')))[0]
            #print(section_header)
            if 'diagnose' in section_header.lower() or 'DIAGNOSE' in section or 'Diagnose :' in section or 'Problemstilling :' in section:
                diagnose_detected = True
            else:
                cleaned_text += section + ' '
    if not diagnose_detected :
        print('No DIAGNOSE in: ', journal_id)
    return cleaned_text

def write_to_disk(parsed_data, pdata_folder, dataset_type, exclude_diagnosis=True):
    start_time = time.time()
    dataset_folder = os.path.join(pdata_folder, dataset_type)
    doc_type = 'inkomst'
    if 'epikrise' in dataset_type.lower():
        doc_type = 'epikrise'
    if dataset_type not in os.listdir(pdata_folder):
        os.mkdir(dataset_folder)
    labels = set(list(parsed_data.loc[:,'Beskrivelse']))
    for label in labels:
        if label == 'Negativ':
            label = 'Not_R55'
        if label not in os.listdir(dataset_folder):
            os.mkdir(os.path.join(dataset_folder, label))
    nr_saved = 0
    for i, row in parsed_data.iterrows():
        
        journal_id = str(row['JOURNALID']).split('.')[0]
        #proc_text = row['JournalTextUDPipe']
        proc_text = row['JournalTextUDPipe_newpar']
        label = row['Beskrivelse']
        if label == 'Negativ':
            label = 'Not_R55'
        class_folder = os.path.join(dataset_folder, label)
        file_name = journal_id + '.txt' # handles Pandas conversion of IDs to float (xxx.0)
        if file_name not in os.listdir(class_folder):
            with codecs.open(os.path.join(class_folder, file_name), 'w', 'utf-8') as f:
                try:
                    if exclude_diagnosis:
                        proc_text = remove_info(proc_text, journal_id, label, doc_type)
                    if proc_text:
                        f.write(proc_text)
                        nr_saved += 1
                        if nr_saved % 100 == 0:
                            print('Saved %d rows' % nr_saved)
                    else:
                        f.close()
                        os.remove(os.path.join(class_folder, file_name))
                except UnicodeEncodeError:
                    print(file_name)        # JOURNALID = 66476665 (Epi)
        else:
            print('File already exists and will not be overwritten. Delete / rename it and re-run.')
    print('Finished saving data in "%s"' % dataset_folder)
    SluttTid = time.time()
    print (round(SluttTid - start_time, 2), "Sekunder")
    

#########################
# Stats on extracted data
#########################

def read_texts(pdata_folder, dataset_type):
    """ Returns a list of tuples with (text, label, text_id).
    """
    texts = []
    for class_folder in  os.listdir(os.path.join(pdata_folder, dataset_type)):
        for file_name in os.listdir(os.path.join(pdata_folder, dataset_type, class_folder)):
            with codecs.open(os.path.join(pdata_folder, dataset_type, class_folder, file_name), 
                             'r', 'utf-8') as f:
                texts.append((f.read(), class_folder, file_name[:-4]))
    return texts

def find_short(text_lens):
    short_docs = []
    for label, lens_per_lbl in text_lens.items():
        for text_id, text_len in lens_per_lbl:
            if text_len < 100:
                short_docs.append((text_id, text_len))
    print('Nr of short (<100 w) texts: {}'.format(len(short_docs)))
    print('\t {:<15}{}'.format('Short text id', 'Len'))
    for short_doc in short_docs:
        print('\t {:<15}{}'.format(short_doc[0], short_doc[1]))
    print()

def get_len(texts):
    lens = {'R55':[], 'Pre_R55':[], 'Not_R55':[]}
    for text, label, text_id in texts:
        words = text.split()
        lens[label].append((text_id,len(words))) 
    all_lens = []
    print('Statistics about text length (# words)')
    print('{:<8}{:<8} ({:<5})'.format('Label', 'Avg len', 'St. dev.'))
    for label, lens_per_lbl in lens.items():
        len_only = [el[1] for el in lens_per_lbl]
        print('{:<8}{:<8.2f} ({:<5.2f})'.format(label, np.mean(len_only), np.std(len_only)))
        all_lens.extend(len_only)
    print('{:<8}{:<8.2f} ({:<5.2f})'.format('All', np.mean(all_lens), np.std(all_lens)))
    print()
    return lens

def get_mentions(texts, word):
    """ Returns the ids of the documents mentioning 
    'word' at least once.
    """
    mentions = {}
    for text, label, text_id in texts:
        if word in text.lower():
            if label not in mentions:
                mentions[label] = [text_id]
            else:
                if text_id not in mentions[label]:
                    mentions[label].append(text_id)
    return mentions

def has_mention(text, target_strings):
    for ts in target_strings:
        if ts in text:
            return True

def count_dept(texts, labels):
    """ Counts departments.
    """
    dept_cnt = {}
    for label in labels:
        dept_cnt[label] = Counter()
    nr_skipped = 0
    if len(texts[0]) > 3:
        texts = [(text_id, text, label) for (text_id, text, label, orig_label) in texts]
    for text_id, text, label in texts:
        try:
            dept = re.match('.* (?=INNKOMSTJOURNAL)', text)[0].split()[1]
            dept_cnt[label][dept] += 1
        except:
            nr_skipped += 1
            print("Can't locate department info for {}".format(text_id))
    print('Skipped from count: ', nr_skipped)
    for label in dept_cnt:
        print('\t Dept for {}'.format(label))
        for dept in sorted(dept_cnt[label], key=lambda x: x[1], reverse=True):
            print('\t{:<7}{}'.format(dept_cnt[label][dept], dept))
    
    return dept_cnt
