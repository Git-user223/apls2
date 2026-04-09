#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create binary raster masks from SpaceNet road vector GeoJSONs.

Modernized from the original CosmiQ Works apls package:
  - Replaced sys.path hacks with relative package import
  - Fixed deprecated pandas .ix[] → .iloc[]
  - Updated convert_to_8Bit function name (was convert_to_8Bit in original)
"""

import os
import time
import argparse
import pandas as pd

from . import apls_utils


###############################################################################
def create_masks(path_data, buffer_meters=2, n_bands=3,
                 burnValue=150, make_plots=True, overwrite_ims=False,
                 output_df_file='',
                 header=None):
    """
    Create 8-bit images and binary road masks for all GeoTIFFs in path_data.

    Writes converted 8-bit images and road masks to subdirectories alongside
    the source imagery. Returns a DataFrame of file locations.

    Arguments
    ---------
    path_data : str
        Root directory containing ``geojson/spacenetroads/``, imagery
        subdirectories, etc.
    buffer_meters : float
        Road buffer radius in metres. Defaults to ``2``.
    n_bands : int
        3 (RGB) or 8 (multispectral). Defaults to ``3``.
    burnValue : int
        Pixel value for road pixels in the mask. Defaults to ``150``.
    make_plots : bool
        Save PNG visualisations alongside masks. Defaults to ``True``.
    overwrite_ims : bool
        Re-create outputs even if they already exist. Defaults to ``False``.
    output_df_file : str
        Path to write the output CSV. Defaults to ``''`` (no file written).
    header : list or None
        Column names for the output DataFrame. Defaults to the standard
        5-column header.

    Returns
    -------
    df : pd.DataFrame
        Columns: name, im_file, im_vis_file, mask_file, mask_vis_file.
    """
    if header is None:
        header = ['name', 'im_file', 'im_vis_file', 'mask_file',
                  'mask_vis_file']

    t0 = time.time()

    path_labels = os.path.join(path_data, 'geojson', 'spacenetroads')
    path_masks = os.path.join(path_data, 'masks_' + str(buffer_meters) + 'm')
    path_masks_plot = os.path.join(
        path_data, 'masks_' + str(buffer_meters) + 'm_plots')
    path_images_vis = os.path.join(path_data, 'RGB-PanSharpen_8bit')

    if n_bands == 3:
        path_images_raw = os.path.join(path_data, 'RGB-PanSharpen')
        path_images_8bit = os.path.join(path_data, 'RGB-PanSharpen_8bit')
    else:
        path_images_raw = os.path.join(path_data, 'MUL-PanSharpen')
        path_images_8bit = os.path.join(path_data, 'MUL-PanSharpen_8bit')
        if not os.path.exists(path_images_vis):
            print("Need to run 3-band pass before 8-band!")
            return None

    for d in [path_images_8bit, path_masks, path_masks_plot]:
        os.makedirs(d, exist_ok=True)

    outfile_list = []
    im_files = os.listdir(path_images_raw)
    nfiles = len(im_files)

    for i, im_name in enumerate(im_files):
        if not im_name.endswith('.tif'):
            continue

        name_root = 'AOI' + im_name.split('AOI')[1].split('.')[0]
        im_file_raw = os.path.join(path_images_raw, im_name)
        im_file_out = os.path.join(path_images_8bit, im_name)
        im_file_out_vis = im_file_out.replace('MUL', 'RGB')

        if not os.path.exists(im_file_out) or overwrite_ims:
            apls_utils.convertTo8Bit(im_file_raw, im_file_out,
                                     outputPixType='Byte',
                                     outputFormat='GTiff',
                                     rescale_type='rescale',
                                     percentiles=[2, 98])

        label_file = os.path.join(
            path_labels, 'spacenetroads_' + name_root + '.geojson')
        mask_file = os.path.join(path_masks, name_root + '.png')
        plot_file = os.path.join(path_masks_plot, name_root + '.png') \
            if make_plots else ''

        print("\n", i+1, "/", nfiles)
        print("  im_name:", im_name)
        print("  name_root:", name_root)
        print("  im_file_out:", im_file_out)
        print("  mask_file:", mask_file)
        print("  plot_file:", plot_file)

        if not os.path.exists(mask_file) or overwrite_ims:
            apls_utils._get_road_buffer(label_file,
                                        im_file_out_vis,
                                        mask_file,
                                        buffer_meters=buffer_meters,
                                        burnValue=burnValue,
                                        bufferRoundness=6,
                                        plot_file=plot_file,
                                        figsize=(6, 6),
                                        fontsize=8,
                                        dpi=500,
                                        show_plot=False,
                                        verbose=False)

        outfile_list.append([im_name, im_file_out, im_file_out_vis,
                              mask_file, mask_file])

    df = pd.DataFrame(outfile_list, columns=header)
    if output_df_file:
        df.to_csv(output_df_file, index=False)

    print("\ndf.iloc[0]:", df.iloc[0])
    print("Total data length:", len(df))
    print("Time to run create_masks():", time.time() - t0, "seconds")
    return df


###############################################################################
def main():
    """CLI entry point for create_spacenet_masks."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path_data',
        default='/spacenet_data/sample_data/AOI_2_Vegas_Train',
        help='Folder containing imagery and geojson labels')
    parser.add_argument(
        '--output_df_path',
        default='/spacenet_data/sample_data',
        help='Directory for the output CSV')
    parser.add_argument('--buffer_meters', default=2, type=float)
    parser.add_argument('--n_bands', default=3, type=int,
                        help='Number of bands [3, 8]')
    parser.add_argument('--burnValue', default=150, type=int)
    parser.add_argument('--make_plots', default=1, type=int)
    parser.add_argument('--overwrite_ims', default=1, type=int)

    args = parser.parse_args()

    data_root = 'AOI' + args.path_data.split('AOI')[-1].replace('/', '_')
    output_df_file = os.path.join(
        args.output_df_path,
        data_root + '_files_loc_' + str(args.buffer_meters) + 'm.csv')

    df = create_masks(
        args.path_data,
        buffer_meters=args.buffer_meters,
        n_bands=args.n_bands,
        burnValue=args.burnValue,
        output_df_file=output_df_file,
        make_plots=bool(args.make_plots),
        overwrite_ims=bool(args.overwrite_ims))

    print("Output CSV:", output_df_file)
    return df


###############################################################################
if __name__ == "__main__":
    main()
