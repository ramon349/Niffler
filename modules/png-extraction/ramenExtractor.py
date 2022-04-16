#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import glob 
from shutil import copyfile
import hashlib
import json
import sys
import subprocess
import logging
from multiprocessing import Pool
import pdb
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import pydicom as dicom 
import png
# pydicom imports needed to handle data errors
from pydicom import config
from pydicom import datadict
from pydicom import values 

import pathlib
from extractUtils import extraction_main ,get_dcms
configs = {}

#new imports 

def initialize_config_and_execute(config_values):
    global configs
    configs = config_values
    # Applying checks for paths
    
    p1 = pathlib.PurePath(configs['DICOMHome'])
    dicom_home = p1.as_posix() # the folder containing your dicom files

    p2 = pathlib.PurePath(configs['OutputDirectory'])
    output_directory = p2.as_posix()

    print_images = bool(configs['PrintImages'])
    print_only_common_headers = bool(configs['CommonHeadersOnly'])
    processes = int(configs['UseProcesses']) # how many processes to use.
    email = configs['YourEmail']
    send_email = bool(configs['SendEmail'])
    no_splits = int(configs['SplitIntoChunks'])
    is16Bit = bool(configs['is16Bit']) 
    
    metadata_col_freq_threshold = 0.1

    png_destination = output_directory + '/extracted-images/'
    failed = output_directory + '/failed-dicom/'
    maps_directory = output_directory + '/maps/'
    meta_directory = output_directory + '/meta/'

    LOG_FILENAME = output_directory + '/ImageExtractor.out'
    pickle_file = output_directory + '/ImageExtractor.pickle'

    # record the start time
    t_start = time.time()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

    if not os.path.exists(maps_directory):
        os.makedirs(maps_directory)

    if not os.path.exists(meta_directory):
        os.makedirs(meta_directory)

    if not os.path.exists(png_destination):
        os.makedirs(png_destination)

    if not os.path.exists(failed):
        os.makedirs(failed)

    if not os.path.exists(failed + "/1"):
        os.makedirs(failed + "/1")

    if not os.path.exists(failed + "/2"):
        os.makedirs(failed + "/2")

    if not os.path.exists(failed + "/3"):
        os.makedirs(failed + "/3")

    if not os.path.exists(failed + "/4"):
        os.makedirs(failed + "/4")

    logging.info("------- Values Initialization DONE -------")
    final_res = execute(pickle_file, dicom_home, output_directory, print_images, print_only_common_headers,
                        processes,  email, send_email, no_splits, is16Bit, png_destination,
        failed, maps_directory, meta_directory, LOG_FILENAME, metadata_col_freq_threshold, t_start)
    return final_res


# Function when pydicom fails to read a value attempt to read as other types.
def fix_mismatch_callback(raw_elem, **kwargs):
    try:
        if raw_elem.VR: 
            values.convert_value(raw_elem.VR, raw_elem)
    except TypeError as err:
        logging.error(err)
    except BaseException as err:
        for vr in kwargs['with_VRs']:
            try:
                values.convert_value(vr, raw_elem)
            except ValueError:
                pass
            except TypeError:
                continue
            else:
                raw_elem = raw_elem._replace(VR=vr)
    return raw_elem



    
# Function used by pydicom.
def fix_mismatch(with_VRs=['PN', 'DS', 'IS', 'LO', 'OB']):
    """A callback function to check that RawDataElements are translatable
    with their provided VRs.  If not, re-attempt translation using
    some other translators.
    Parameters
    ----------
    with_VRs : list, [['PN', 'DS', 'IS']]
        A list of VR strings to attempt if the raw data element value cannot
        be translated with the raw data element's VR.
    Returns
    -------
    No return value.  The callback function will return either
    the original RawDataElement instance, or one with a fixed VR.
    """
    dicom.config.data_element_callback = fix_mismatch_callback
    config.data_element_callback_kwargs = {
        'with_VRs': with_VRs,
    }    

def execute(pickle_file, dicom_home, output_directory, print_images, print_only_common_headers, 
            processes, email, send_email, no_splits, is16Bit, png_destination,
    failed, maps_directory, meta_directory, LOG_FILENAME, metadata_col_freq_threshold, t_start,group_volumes=False):
    err = None
    fix_mismatch()
    if processes == 0.5:  # use half the cores to avoid  high ram usage
        core_count = int(os.cpu_count()/2)
    elif processes == 0:  # use all the cores
        core_count = int(os.cpu_count())
    elif processes < os.cpu_count():  # use the specified number of cores to avoid high ram usage
        core_count = processes
    else:
        core_count = int(os.cpu_count())
    if os.path.isfile(pickle_file):
        f=open(pickle_file,'rb')
        filelist=pickle.load(f)
    else:
        filelist = get_dcms(dicom_home,group_volumes=group_volumes)
        pickle.dump(filelist,open(pickle_file,'wb'))
    file_chunks = np.array_split(filelist,no_splits)
    logging.info('Number of dicom files: ' + str(len(filelist)))

    try:
        filelist[0] # load first file as a template to look at all
    except IndexError:
        logging.error("There is no file present in the given folder in " + dicom_home)
        sys.exit(1)

    logging.debug('Loaded the first file successfully')
    extract_config = dict()
    extract_config['print_images']=print_images
    extract_config['png_destination'] =png_destination
    num_chunks = len(file_chunks)
    n_chunk = 0
    for i,chunk in enumerate(file_chunks):
        csv_destination = "{}/meta/metadata_{}.csv".format(output_directory,i)
        mappings = "{}/maps/mapping_{}.csv".format(output_directory,i)
        fm = open(mappings, "w+")
        filemapping = 'Original DICOM file location, PNG location \n'
        fm.write(filemapping)

        # add a check to see if the metadata has already been extracted
        # step through whole file list, read in file, append fields to future dataframe of all files

        headerlist = []
        # start up a multi processing pool
        # for every item in filelist send data to a subprocess and run extract_headers func
        # output is then added to headerlist as they are completed (no ordering is done)
        seg = [(row_id,row_path,extract_config) for row_id,row_path in enumerate(chunk) ]
        total_rows = len(chunk) 
        proc_counts = 0 
        with Pool(core_count) as p:
            res= p.imap_unordered(extraction_main,seg)  
            for i,e in enumerate(res):
                headerlist.append(e)
                print(f"Finished Chunk {proc_counts} out of {total_rows}   on File_group: {n_chunk}:{num_chunks}",end='\r')
                proc_counts +=1 
        data = pd.DataFrame(headerlist)
        n_chunk +=1 
        logging.info('Chunk ' + str(i) + ' Number of fields per file : ' + str(len(data.columns)))
        # find common fields
        # make dataframe containing all fields and all files minus those removed in previous block
        # export csv file of final dataframe
        data.to_csv(csv_destination, index = None, header=True)
        logging.info('Chunk run time: %s %s', time.time() - t_start, ' seconds!')
    logging.info('Generating final metadata file')
    col_names = dict()
    all_headers = dict()
    total_length = 0
    metas = glob.glob( "{}*.csv".format(meta_directory))
    # for each meta  file identify the columns that are not na's for at least 10% (metadata_col_freq_threshold) of data
    for meta in metas:
        m = pd.read_csv(meta,dtype='str')
        d_len = m.shape[0]
        total_length += d_len
        for e in m.columns:
            col_pop = d_len - np.sum(m[e].isna()) # number of populated rows for this column in this metadata file
            if e in col_names:
                col_names[e] += col_pop
            else:
                col_names[e] = col_pop
            # all_headers keeps track of number of appearances of each header. We later use this count to ensure that
            # the headers we use are present in all metadata files.
            if e in all_headers:
                all_headers[e] += 1
            else:
                all_headers[e] = 1
    loadable_names = list()
    for k in col_names.keys():
        if k in all_headers and all_headers[k] >= no_splits:  # no_splits == number of batches used 
            if col_names[k] >= metadata_col_freq_threshold*total_length:
                loadable_names.append(k) # use header only if it's present in every metadata file
    # load every metadata file using only valid columns
    meta_list = list()
    for meta in metas:
        m = pd.read_csv(meta,dtype='str',usecols=loadable_names)
        meta_list.append(m)
    merged_meta = pd.concat(meta_list,ignore_index=True)
    merged_meta.to_csv('{}/metadata.csv'.format(output_directory),index=False)

    if send_email:
       subprocess.call('echo "Niffler has successfully completed the png conversion" | mail -s "The image conversion'
                       ' has been complete" {0}'.format(email), shell=True)
    # Record the total run-time
    logging.info('Total run time: %s %s', time.time() - t_start, ' seconds!')
    logging.shutdown()  # Closing logging file after extraction is done !!
    logs = []
    logs.append(err)
    logs.append("The PNG conversion is SUCCESSFUL")
    return logs

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        niffler = json.load(f)
    # CLI Argument Parser
    ap = argparse.ArgumentParser()
    ap.add_argument("--DICOMHome", default=niffler['DICOMHome'])
    ap.add_argument("--OutputDirectory", default=niffler['OutputDirectory'])
    ap.add_argument("--SplitIntoChunks", default=niffler['SplitIntoChunks'])
    ap.add_argument("--PrintImages", default=niffler['PrintImages'])
    ap.add_argument("--CommonHeadersOnly", default=niffler['CommonHeadersOnly'])
    ap.add_argument("--UseProcesses", default=niffler['UseProcesses'])
    ap.add_argument("--is16Bit", default=niffler['is16Bit'])
    ap.add_argument("--SendEmail", default=niffler['SendEmail'])
    ap.add_argument("--YourEmail", default=niffler['YourEmail'])
    args = vars(ap.parse_args())
    if len(args) > 0:
        initialize_config_and_execute(args)
    else:
        initialize_config_and_execute(niffler)
