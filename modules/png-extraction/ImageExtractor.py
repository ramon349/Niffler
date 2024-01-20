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
from multiprocessing.pool import ThreadPool as Pool
import pdb
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import pydicom as pyd 
import png
# pydicom imports needed to handle data errors
from pydicom import config
from pydicom import values
import sys 
import pathlib
configs = {}
"""
Reasonings 
- removing the depth variable as we should be able to extract all dcm files in a directory. 
- any trestriction should be dealt by the user no the library 
- we remove the CommonHeadersOnly as that may be a limitation from the  public vs private header debate  
- Flattening to a certain level is also remove. The groundtruth will be your metadata  and mapping file only use that 
- if code requires something else then gg's 
- email support will be added at a later time 
- images should also be stored in whatever depth encoding they are 
- we should store images in their original depth. forcing 8 bit to be 16 bit arbitrarly is odd 
- TODO: We should ad an option to apply voi and LUT functiosn if needed 
- alll path construction is done through joins so correct path seprators are used idk why worry about windows but here we are
- Who uses the start time as a variable that is then passed as a parameter why was this a good idea 
- People should be capable enough to figure out how many proceses they can use. We can set a default of 4 in case they have no idea 
- Using 0.5 is odd choice 
-refernece of depth  variable remove. just do a full search 
- TODO: we should honestly just have a processing queue and have worker proceses doing he extraction and writing. 
- Non need for chunking do everything in real time 
- i already use imap processign i could return the metadata file and the paths image extractd. Then i can save every n batch of extraction 
- MOVING to just using chunking 
- TODO: With the dicom tag processing what was the need for looking into sequence and evaluation. might be worth using the new approach i tried 
"""
def populate_extraction_dirs(output_directory): 
    png_destination = os.path.join(output_directory,'extracted-images')
    failed = os.path.join(output_directory ,'failed-dicom')
    maps_directory = os.path.join(output_directory ,'maps')
    meta_directory = os.path.join(output_directory ,'meta')
    if not os.path.exists(maps_directory):
        os.makedirs(maps_directory)

    if not os.path.exists(meta_directory):
        os.makedirs(meta_directory)

    if not os.path.exists(png_destination):
        os.makedirs(png_destination)

    if not os.path.exists(failed):
        os.makedirs(failed)

    for e in range(6): 
        fail_dir = os.path.join(failed,str(e)) 
        if not os.path.exists(fail_dir):
            os.makedirs(fail_dir)
    print(f"Done Creating Directories")

def initialize_config_and_execute(config_values):
    global configs
    configs = config_values
    # Applying checks for paths

    dicom_home = str(pathlib.PurePath(configs['DICOMHome'])) #parse the path and convert it to a string 
    output_directory = str(pathlib.Path(configs['OutputDirectory']))
    #output_directory = p2.as_posix() # TODO: I donot udnerstand why this is needed 

    print_images = configs['SavePNGs']
    PublicHeadersOnly = configs['PublicHeadersOnly']
    processes =  configs['NumProcesses']
    SaveBatchSize = configs['SaveBatchSize']

    png_destination = os.path.join(output_directory,'extracted-images')
    failed = os.path.join(output_directory ,'failed-dicom')
    maps_directory = os.path.join(output_directory ,'maps')
    meta_directory = os.path.join(output_directory ,'meta')

    LOG_FILENAME = os.path.join(output_directory ,'ImageExtractor.out')
    pickle_file = os.path.join(output_directory ,'ImageExtractor.pickle') #TODO: i forgot what this does 
    populate_extraction_dirs(output_directory=output_directory)

    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)


    logging.info("------- Values Initialization DONE -------")
    final_res = execute(pickle_file=pickle_file,dicom_home=dicom_home,
    output_directory=output_directory,print_images=print_images,
    processes=processes,SaveBatchSize=SaveBatchSize,
    png_destination=png_destination,failed=failed,maps_directory=maps_directory,
    meta_directory=meta_directory,LOG_FILENAME=LOG_FILENAME,PublicHeadersOnly=PublicHeadersOnly)
    return final_res



# Function for getting tuple for field,val pairs
def get_tuples(plan, PublicHeadersOnly, outlist=None, key=""): 
    x = plan.keys()
    x = list(x)
    for i in x:
        if not i.is_private or PublicHeadersOnly==False:
            outlist.append((plan[i].name, plan[i].value))
    return outlist


def extract_dcm(plan,dcm_path,PublicHeadersOnly=None,SaveImages=None,PngDestination=None,OutputDirectory=None,FailDirectory=None):

    # checks all dicom fields to make sure they are valid
    # if an error occurs, will delete it from the data structure
    dcm_dict_copy = list(plan._dict.keys())

    for tag in dcm_dict_copy:
        try:
            plan[tag]
        except:
            logging.warning("dropped fatal DICOM tag {}".format(tag))
            del plan[tag]
    c = True
    try:
        check = plan.pixel_array  # throws error if dicom file has no image
        extract_images(plan,png_destination=PngDestination,failed=FailDirectory)
    except:
        c = False
    kv = get_tuples(plan, PublicHeadersOnly)  # gets tuple for field,val pairs for this file. function defined above
    dicom_tags_limit = 1000  #TODO: i should add this as some sort of extra limit in the config 
    if len(kv) > dicom_tags_limit:
        logging.debug(str(len(kv)) + " dicom tags produced by " + dcm_path)
        copyfile(dcm_path,os.path.join(OutputDirectory,'failed-dicom',str(5),+ os.path.basename(dcm_path)) )
    kv.append(('file', dcm_path))  # adds my custom field with the original filepath
    kv.append(('has_pix_array', c))  # adds my custom field with if file has image
    if c:
        # adds my custom category field - useful if classifying images before processing
        kv.append(('category', 'uncategorized'))
    else:
        kv.append(('category', 'no image'))  # adds my custom category field, makes note as imageless
    return dict(kv)


def rgb_store_format(arr):
    """ Create a  list containing pixels  in format expected by pypng 
    arr: numpy array to be modified. 

    We create an array such that an  nxmx3  matrix becomes a list of n elements. 
    Each element contains m*3 items. 
    """
    out= list(arr) 
    flat_out = list() 
    for e in out:
        flat_out.append(list())
        for k in e: 
            flat_out[-1].extend(k)
    return flat_out 

# Function to extract pixel array information
# takes an integer used to index into the global filedata dataframe
# returns tuple of
# filemapping: dicom to png paths   (as str)
# fail_path: dicom to failed folder (as tuple)
# found_err: error code produced when processing
def extract_images(ds, png_destination, flattened_to_level, failed, is16Bit):
    found_err = None
    filemapping = ""
    fail_path = ""
    pdb.set_trace()
    try:
        im = ds.pixel_array  # pull image from read dicom
        ID1 = ds.PatientID.value
        try: 
            ID2 = ds.StudyInstanceUID.value
        except: 
            ID2 = "ALL-STUDIES" 
        try:
            ID3 = ds.SeriesInstanceUID.value 
        except: 
            ID3= "ALL-SERIES"
        folderName = hashlib.sha224(ID1.encode('utf-8')).hexdigest() + "/" + \
                        hashlib.sha224(ID2.encode('utf-8')).hexdigest()

        img_name = hashlib.sha224(ID3.encode('utf-8')).hexdigest() +'.png'
         
        # check for existence of the folder tree patient/study/series. Create if it does not exist.
        os.makedirs(png_destination + folderName, exist_ok=True)

        pngfile = os.path.join(png_destination,folderName , img_name+ '.png')
        try: 
            isRGB = (ds.PhotometricInterpretation.value =='RGB')
        except: 
            isRGB = False  
        if is16Bit:
            # write the PNG file as a 16-bit greyscale 
            image_2d = ds.pixel_array.astype(np.double)
            # # Rescaling grey scale between 0-255
            image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 65535.0
            # # Convert to uint
            shape = ds.pixel_array.shape
            image_2d_scaled = np.uint16(image_2d_scaled)
            with open(pngfile, 'wb') as png_file:
                if isRGB:
                    image_2d_scaled = rgb_store_format(image_2d_scaled)
                    w = png.Writer(shape[1], shape[0], greyscale=False, bitdepth=16)
                else:
                    w = png.Writer(shape[1], shape[0], greyscale=True, bitdepth=16)
                w.write(png_file, image_2d_scaled)
        else:
            shape = ds.pixel_array.shape
            # Convert to float to avoid overflow or underflow losses.
            image_2d = ds.pixel_array.astype(float)
            # Rescaling grey scale between 0-255
            image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
            # onvert to uint
            image_2d_scaled = np.uint8(image_2d_scaled)
            # Write the PNG file
            with open(pngfile, 'wb') as png_file:
                if isRGB:
                    image_2d_scaled = rgb_store_format(image_2d_scaled)
                    w = png.Writer(shape[1], shape[0], greyscale=False)
                else:
                    w = png.Writer(shape[1], shape[0], greyscale=True)
                w.write(png_file, image_2d_scaled)
        filemapping = filedata.iloc[i].loc['file'] + ', ' + pngfile + '\n'
    except AttributeError as error:
        found_err = error
        logging.error(found_err)
        fail_path = filedata.iloc[i].loc['file'], failed + '1/' + \
                    os.path.split(filedata.iloc[i].loc['file'])[1][:-4] + '.dcm'
    except ValueError as error:
        found_err = error
        logging.error(found_err)
        fail_path = filedata.iloc[i].loc['file'], failed + '2/' + \
                    os.path.split(filedata.iloc[i].loc['file'])[1][:-4] + '.dcm'
    except BaseException as error:
        found_err = error
        logging.error(found_err)
        fail_path = filedata.iloc[i].loc['file'], failed + '3/' + \
                    os.path.split(filedata.iloc[i].loc['file'])[1][:-4] + '.dcm'
    except Exception as error:
        found_err = error
        logging.error(found_err)
        fail_path = filedata.iloc[i].loc['file'], failed + '4/' + \
                    os.path.split(filedata.iloc[i].loc['file'])[1][:-4] + '.dcm'
    return (filemapping, fail_path, found_err)


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


from pathlib import Path
def proper_extraction(dcm_path,png_destination): 
    dcm = pyd.dcmread(dcm_path) 
    try: 
        dicom_tags = extract_headers() 
    except: 
        return  None  #figure out the paths for failure
    try: 
        (png_path )= extract_images()

def execute(pickle_file=None, dicom_home=None, output_directory=None, print_images=None,
            processes=None,  SaveBatchSize=None,  png_destination=None,
            failed=None, maps_directory=None, meta_directory=None, LOG_FILENAME=None,PublicHeadersOnly=None):
    err = None
    fix_mismatch() #TODO: Please check exactly what this might fix  
    core_count = processes
    # gets all dicom files. if editing this code, get filelist into the format of a list of strings,
    # with each string as the file path to a different dicom file.
    if os.path.isfile(pickle_file):
        f = open(pickle_file, 'rb')
        filelist = pickle.load(f)
    else:
        print(dicom_home)
        print("Getting all the dcms in your project. May take a while :)")
        filelist = [str(e) for e in Path(dicom_home).rglob("*.dcm") ] 
        pickle.dump(filelist, open(pickle_file, 'wb'))
    # if the number of files is less than the specified number of splits, then
        
    logging.info('Number of dicom files: ' + str(len(filelist)))

    try:
        ff = filelist[0]  # load first file as a template to look at all
    except IndexError:
        logging.error("There is no file present in the given folder in " + dicom_home)
        sys.exit(1)

    plan = dicom.dcmread(ff, force=True)
    logging.debug('Loaded the first file successfully')

    keys = [(aa) for aa in plan.dir() if (hasattr(plan, aa) and aa != 'PixelData')]
    # checks for images in fields and prints where they are #TODO: WHY DOES THIS EXIST 
    for field in plan.dir():
        if (hasattr(plan, field) and field != 'PixelData'):
            entry = getattr(plan, field)
            if type(entry) is bytes:
                logging.debug(field)
                logging.debug(str(entry))
    file_map_list = list() 
    meta_df_list = list()
    pdb.set_trace()
    chunk_timestamp = time.time()
    chunks = np.array_split(filelist,10)
    outs = extract_headers((0,filelist[0],True,'./this/'))
    pdb.set_trace() 
    pdb.set_trace()

    chunks_list = [tups + (PublicHeadersOnly,) + (output_directory,) for tups in enumerate(chunk)]
    with Pool(core_count) as p:
        for extract_out in p.imap_unordered(extract_image,filelist):
            (fmap, fail_path,meta,err) = extract_out 
            if  not err: 
                meta_df_list.append(meta)
                file_map_list.append(fmap)
            if len(file_map_list) >= SaveBatchSize: 
                #use a function to write a metadata batch file 
                pass 
            csv_destination = "{}/meta/metadata_{}.csv".format(output_directory, i)
            mappings = "{}/maps/mapping_{}.csv".format(output_directory, i)
            fm = open(mappings, "w+")
            filemapping = 'Original DICOM file location, PNG location \n'
            fm.write(filemapping)
        headerlist = []
        # start up a multi processing pool
        # for every item in filelist send data to a subprocess and run extract_headers func
        # output is then added to headerlist as they are completed (no ordering is done)


        with Pool(core_count) as p:
            # we send here print_only_public_headers bool value
            chunks_list = [tups + (PublicHeadersOnly,) + (output_directory,) for tups in enumerate(chunk)]
            res = p.imap_unordered(extract_headers, chunks_list)
            for i, e in enumerate(res):
                headerlist.append(e)
        data = pd.DataFrame(headerlist)
        logging.info('Chunk ' + str(i) + ' Number of fields per file : ' + str(len(data.columns)))
        # export csv file of final dataframe
        if (SpecificHeadersOnly):
            try:
                feature_list = open("featureset.txt").read().splitlines()
                features = []
                for j in feature_list:
                    if j in data.columns:
                        features.append(j)
                meta_data = data[features]
            except:
                meta_data = data
                logging.error("featureset.txt not found")
        else:
            meta_data = data

        fields = data.keys()
        export_csv = meta_data.to_csv(csv_destination, index=None, header=True)
        count = 0  # potential painpoint
        # writting of log handled by main process
        if print_images:
            logging.info("Start processing Images")
            filedata = data
            total = len(chunk)
            stamp = time.time()
            for i in range(len(filedata)):
                if (filedata.iloc[i].loc['file'] is not np.nan):
                    (fmap, fail_path, err) = extract_images(filedata, i, png_destination, flattened_to_level, failed,
                                                            is16Bit)
                    if err:
                        count += 1
                        copyfile(fail_path[0], fail_path[1])
                        err_msg = str(count) + ' out of ' + str(len(chunk)) + ' dicom images have failed extraction'
                        logging.error(err_msg)
                    else:
                        fm.write(fmap)
        fm.close()
        logging.info('Chunk run time: %s %s', time.time() - chunk_timestamp, ' seconds!')

    logging.info('Generating final metadata file')

    col_names = dict()
    all_headers = dict()
    total_length = 0

    metas = glob.glob("{}*.csv".format(meta_directory))
    # for each meta  file identify the columns that are not na's for at least 10% (metadata_col_freq_threshold) of data
    if print_only_common_headers:
        for meta in metas:
            m = pd.read_csv(meta, dtype='str')
            d_len = m.shape[0]
            total_length += d_len

            for e in m.columns:
                col_pop = d_len - np.sum(m[e].isna())  # number of populated rows for this column in this metadata file

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
                if col_names[k] >= metadata_col_freq_threshold * total_length:
                    loadable_names.append(k)  # use header only if it's present in every metadata file

        # load every metadata file using only valid columns
        meta_list = list()
        for meta in metas:
            m = pd.read_csv(meta, dtype='str', usecols=loadable_names)
            meta_list.append(m)
        merged_meta = pd.concat(meta_list, ignore_index=True)
    else:
        # merging_meta
        merged_meta = pd.DataFrame()
        for meta in metas:
            m = pd.read_csv(meta, dtype='str')
            merged_meta = pd.concat([merged_meta, m], ignore_index=True)
    # for only common header
    if print_only_common_headers:
        mask_common_fields = merged_meta.isnull().mean() < 0.1
        common_fields = list(np.asarray(merged_meta.columns)[mask_common_fields])
        merged_meta = merged_meta[common_fields]

    merged_meta.to_csv('{}/metadata.csv'.format(output_directory), index=False)
    # getting a single mapping file
    logging.info('Generating final mapping file')
    mappings = glob.glob("{}/maps/*.csv".format(output_directory))
    map_list = list()
    for mapping in mappings:
        map_list.append(pd.read_csv(mapping, dtype='str'))
    merged_maps = pd.concat(map_list, ignore_index=True)

    merged_maps.to_csv('{}/mapping.csv'.format(output_directory), index=False)

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


def main():
    from configs import get_params 
    config = get_params()  
    initialize_config_and_execute(config)
if __name__ == "__main__":
    main() 