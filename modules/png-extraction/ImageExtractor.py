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
from pydicom.pixel_data_handlers import apply_voi_lut
# pydicom imports needed to handle data errors
from pydicom import config
from pydicom import values
import sys
import pathlib
from pathlib import Path
from functools import partial
from configs import get_params

configs = {}  # TODO:  WHY IS THIS GLOBAL
"""
Reasonings 
- removing the depth variable as we should be able to extract all dcm files in a directory. 
- any trestriction should be dealt by the user no the library 
- we remove the CommonHeadersOnly temporarly. Public vs private headers provides an idirect fix. 
- Flattening to a certain level is also remove. Navigating files should be based on metadatafile.
- TODO: email support will be re-introduced later 
- TODO:  images should be stored in whatever depth encoding they are
- TODO: we should store images in their original depth. forcing 8 bit to be 16 bit arbitrarly is odd 
- TODO: We should ad an option to apply voi and LUT functiosn especially if using PNG extraction  
- TODO: we should honestly just have a processing queue and have worker proceses doing he extraction and writing. 
- TODO: With the dicom tag processing what was the need for looking into sequence and evaluation. might be worth using the new approach i tried 
- TODO: Remove mention of maps directory it is no longer used 
- TODO: Addd some salt parameter to the image naming process. Store it in the image extracter log for backtracking
- TODO: have the fileliest loading check the existing metadata files and filter based on already extracted images 
"""


def populate_extraction_dirs(output_directory: str):
    """
    Images and metadata will be stored in output_directory. This function defines subfolders such as
    metadata and failed dicom folders
    output_directory: str   absolute path to the folder meant to store output artiacts (logs,images,metadata)
    """
    png_destination = os.path.join(output_directory, "extracted-images")
    failed = os.path.join(output_directory, "failed-dicom")
    maps_directory = os.path.join(output_directory, "maps")
    meta_directory = os.path.join(output_directory, "meta")
    if not os.path.exists(maps_directory):
        os.makedirs(maps_directory)

    if not os.path.exists(meta_directory):
        os.makedirs(meta_directory)

    if not os.path.exists(png_destination):
        os.makedirs(png_destination)

    if not os.path.exists(failed):
        os.makedirs(failed)

    for e in range(6):
        fail_dir = os.path.join(failed, str(e))
        if not os.path.exists(fail_dir):
            os.makedirs(fail_dir)
    print(f"Done Creating Directories")


def initialize_config_and_execute(config_values: dict):
    """
    Does all the set up regarding folder setup
    """
    configs = config_values
    # Applying checks for paths

    dicom_home = str(
        pathlib.PurePath(configs["DICOMHome"])
    )  # parse the path and convert it to a string
    output_directory = str(pathlib.Path(configs["OutputDirectory"]))
    # output_directory = p2.as_posix() # TODO: I donot udnerstand why this is needed

    print_images = configs["SavePNGs"]
    PublicHeadersOnly = configs["PublicHeadersOnly"]
    processes = configs["NumProcesses"]
    SaveBatchSize = configs["SaveBatchSize"]

    png_destination = os.path.join(output_directory, "extracted-images")
    failed = os.path.join(output_directory, "failed-dicom")
    meta_directory = os.path.join(output_directory, "meta")

    LOG_FILENAME = os.path.join(output_directory, "ImageExtractor.out")
    pickle_file = os.path.join(
        output_directory, "ImageExtractor.pickle"
    )  # TODO: i forgot what this does
    populate_extraction_dirs(output_directory=output_directory)

    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

    logging.info("------- Values Initialization DONE -------")
    final_res = execute(
        pickle_file=pickle_file,
        dicom_home=dicom_home,
        output_directory=output_directory,
        print_images=print_images,
        processes=processes,
        SaveBatchSize=SaveBatchSize,
        png_destination=png_destination,
        failed=failed,
        MetaDirectory=meta_directory,
        PublicHeadersOnly=PublicHeadersOnly,
    )
    return final_res


# Function for getting tuple for field,val pairs
def get_tuples(plan, PublicHeadersOnly, key=""):
    x = plan.keys()
    x = list(x)
    outlist = list()
    for i in x:
        if not i.is_private or PublicHeadersOnly == False:
            outlist.append((plan[i].name, plan[i].value))
    return outlist


def extract_dcm(
    plan: pyd.Dataset,
    dcm_path: str,
    PublicHeadersOnly: bool = False,
    FailDirectory: str = None,
):
    """ "
    Extract dicom tags from dicom file. Public tags are filtered if specified.
    PNG
    """
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
    except:
        c = False
    kv = get_tuples(
        plan, PublicHeadersOnly
    )  # gets tuple for field,val pairs for this file. function defined above
    dicom_tags_limit = (
        1000  # TODO: i should add this as some sort of extra limit in the config
    )
    if (
        len(kv) > dicom_tags_limit
    ):  # TODO this should not fail silently. What can we do  about it
        logging.debug(str(len(kv)) + " dicom tags produced by " + dcm_path)
        copyfile(
            dcm_path,
            os.path.join(
                FailDirectory, "failed-dicom", str(5), +os.path.basename(dcm_path)
            ),
        )
    kv.append(("file", dcm_path))  # adds my custom field with the original filepath
    kv.append(("has_pix_array", c))  # adds my custom field with if file has image
    if c:
        # adds my custom category field - useful if classifying images before processing
        kv.append(("category", "uncategorized"))
    else:
        kv.append(
            ("category", "no image")
        )  # adds my custom category field, makes note as imageless
    return dict(kv)


def rgb_store_format(arr):
    """Create a  list containing pixels  in format expected by pypng
    arr: numpy array to be modified.

    We create an array such that an  nxmx3  matrix becomes a list of n elements.
    Each element contains m*3 items.
    """
    out = list(arr)
    flat_out = list()
    for e in out:
        flat_out.append(list())
        for k in e:
            flat_out[-1].extend(k)
    return flat_out

def process_image(ds,is16Bit,apply_voi=False,apply_lut=False):
    try:
        isRGB = ds.PhotometricInterpretation.value == "RGB"
    except:
        isRGB = False
    image_2d = ds.pixel_array 
    if apply_voi or apply_lut: 
        image_2d = apply_voi(image_2d,ds,prefer_lut=True)
    image_2d = image_2d.astype(float)
    shape = ds.pixel_array.shape
    if is16Bit:
        # write the PNG file as a 16-bit greyscale
        image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 65535.0
        # # Convert to uint
        image_2d_scaled = np.uint16(image_2d_scaled)
    else:
        # Rescaling grey scale between 0-255
        image_2d_scaled = (np.maximum(image_2d, 0) / image_2d.max()) * 255.0
        # onvert to uint
        image_2d_scaled = np.uint8(image_2d_scaled)
        # Write the PNG file
    return image_2d_scaled,shape,isRGB

# Function to extract pixel array information
# takes an integer used to index into the global filedata dataframe
# returns tuple of
# filemapping: dicom to png paths   (as str)
# fail_path: dicom to failed folder (as tuple)
# found_err: error code produced when processing
def extract_images(ds, png_destination, failed,apply_voi=False,apply_lut=False):
    """
    Function that  extracts a dicom pixel arrayinto a png image. Patient metadata is used to create the file name
    Supports extracting either RGB or Monochrome images. No LUT or VOI is applied at the moment
    Returns
    pngFile --> path to extract png file or None if extraction failed
    err_code --> error code experience dduring extraction. None if no failoure occurs
    """
    err_code = None
    found_err = None
    try:
        ID1 = str(ds.PatientID)
        try:
            ID2 = str(ds.StudyInstanceUID)
        except:
            ID2 = "ALL-STUDIES"
        try:
            ID3 = str(ds.SOPInstanceUID)
        except:
            ID3 = "ALL-SERIES"
        folderName = (
            hashlib.sha224(ID1.encode("utf-8")).hexdigest()
            + "/"
            + hashlib.sha224(ID2.encode("utf-8")).hexdigest()
        )
        img_iden = f"{ID3}"
        img_name = hashlib.sha224(img_iden.encode("utf-8")).hexdigest() + ".png"

        # check for existence of the folder tree patient/study/series. Create if it does not exist.

        store_dir = os.path.join(png_destination, folderName)
        os.makedirs(store_dir, exist_ok=True)
        pngfile = os.path.join(store_dir, img_name)
        image_2d_scaled,shape ,isRGB = process_image(ds,is16Bit=True,apply_lut=apply_lut,apply_voi=apply_voi)
        with open(pngfile, "wb") as png_file:
            if isRGB:
                image_2d_scaled = rgb_store_format(image_2d_scaled)
                w = png.Writer(shape[1], shape[0], greyscale=False)
            else:
                w = png.Writer(shape[1], shape[0], greyscale=True)
            w.write(png_file, image_2d_scaled)
    except AttributeError as error:
        found_err = error
        logging.error(found_err)
        err_code = 1
        pngfile = None
    except ValueError as error:
        found_err = error
        logging.error(found_err)
        err_code = 2
        pngfile = None
    except BaseException as error:
        found_err = error
        logging.error(found_err)
        err_code = 3
        pngfile = None
    return (pngfile, err_code)


# Function when pydicom fails to read a value attempt to read as other types.
def fix_mismatch_callback(raw_elem, **kwargs):
    """
    Specify alternative variable reprepresentations when trying to parse metadata.
    """
    try:
        if raw_elem.VR:
            values.convert_value(raw_elem.VR, raw_elem)
    except TypeError as err:
        logging.error(err)
    except BaseException as err:
        for vr in kwargs["with_VRs"]:
            try:
                values.convert_value(vr, raw_elem)
            except ValueError:
                pass
            except TypeError:
                continue
            else:
                raw_elem = raw_elem._replace(VR=vr)
    return raw_elem


# taken from pydicom docs
def fix_mismatch(with_VRs=["PN", "DS", "IS", "LO", "OB"]):
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
    pyd.config.data_element_callback = fix_mismatch_callback
    config.data_element_callback_kwargs = {
        "with_VRs": with_VRs,
    }


def proper_extraction(
    dcmPath,
    pngDestination: str = None,
    publicHeadersOnly: str = None,
    failDir: str = None,
    apply_voi=False
):
    """
    Run the dicom extraction. We first extract the metadata then we save the image informaiton
    dcm_path: absolute path to a dicom file
    pngDestination: Where to store extracted pngs
    publicHeadersOnly: only use public headers
    """
    dcm = pyd.dcmread(dcmPath, force=True)
    dicom_tags = extract_dcm(dcm, dcm_path=dcmPath, PublicHeadersOnly=publicHeadersOnly)
    if pngDestination and dicom_tags is not None:
        png_path, err_code = extract_images(
            dcm, png_destination=pngDestination, failed=failDir,apply_voi=apply_voi
        )
    else:
        png_path = None
    dicom_tags["png_path"] = png_path
    dicom_tags["err_code"] = err_code
    return dicom_tags


def execute(
    pickle_file=None,
    dicom_home=None,
    output_directory=None,
    print_images=None,
    processes=None,
    SaveBatchSize=None,
    png_destination=None,
    failed=None,
    PublicHeadersOnly=None,
    MetaDirectory=None,
):
    err = None
    fix_mismatch()  # TODO: Please check exactly what this might fix
    core_count = processes
    # gets all dicom files. if editing this code, get filelist into the format of a list of strings,
    # with each string as the file path to a different dicom file.
    if os.path.isfile(pickle_file):
        f = open(pickle_file, "rb")
        filelist = pickle.load(f)
    else:
        print(dicom_home)
        print("Getting all the dcms in your project. May take a while :)")
        filelist = [str(e) for e in Path(dicom_home).rglob("*.dcm")]
        pickle.dump(filelist, open(pickle_file, "wb"))
    # if the number of files is less than the specified number of splits, then

    logging.info("Number of dicom files: " + str(len(filelist)))

    try:
        ff = filelist[0]  # load first file as a template to look at all
    except IndexError:
        logging.error("There is no file present in the given folder in " + dicom_home)
        sys.exit(1)

    logging.debug("Loaded the first file successfully")

    chunk_timestamp = time.time()
    # TODO: if there is a more understandable way of using imap where some parameters that are constant let me know
    # some version of python has it so you can define keyword args will look into later
    extractor = partial(
        proper_extraction,
        pngDestination=png_destination,
        publicHeadersOnly=PublicHeadersOnly,
        failDir=failed,
    )
    meta_rows = list()
    t_start = time.time()
    counter = 0
    with Pool(core_count) as p:
        for i, dcm_meta in enumerate(p.imap_unordered(extractor, filelist)):
            meta_rows.append(dcm_meta)
            if len(meta_rows) >= SaveBatchSize:
                meta_df = pd.DataFrame(meta_rows)
                meta_rows = list()
                csv_destination = f"{output_directory}/meta/metadata_{counter}.csv"
                counter += 1
                meta_df.to_csv(csv_destination)
                logging.info(
                    "Chunk run time: %s %s", time.time() - chunk_timestamp, " seconds!"
                )
    if len(meta_rows) >0: 
        meta_df = pd.DataFrame(meta_rows)
        meta_df.to_csv(csv_destination)
    meta_directory = f"{output_directory}/meta/"
    metas = glob.glob(f"{meta_directory}*.csv")
    merged_meta = pd.DataFrame()
    # TODO:  Right now we do not fillter out empty metadata columsn. add it in the future?
    for meta in metas:
        m = pd.read_csv(meta, dtype="str")
        merged_meta = pd.concat([merged_meta, m], ignore_index=True)
    merged_meta.to_csv("{}/metadata.csv".format(output_directory), index=False)
    logging.info("Total run time: %s %s", time.time() - t_start, " seconds!")
    logging.shutdown()  # Closing logging file after extraction is done !!
    logs = []
    logs.append(err)
    logs.append("The PNG conversion is SUCCESSFUL")
    return logs


def main():
    config = get_params()
    initialize_config_and_execute(config)


if __name__ == "__main__":
    main()
