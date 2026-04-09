#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modernized apls_utils for the apls2 package.

Replaces osgeo (gdal/ogr/osr), affine, cv2, and skimage dependencies with
rasterio, pyproj, and numpy equivalents.
"""

import numpy as np
import rasterio
from rasterio.transform import rowcol, xy as rasterio_xy
from rasterio.features import rasterize as rio_rasterize
from rasterio.transform import Affine
from pyproj import Transformer
import scipy.spatial
import geopandas as gpd
import shapely
import time
import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.colors
from math import sqrt, radians, cos, sin, asin

from . import osmnx_funcs


###############################################################################
def pixelToGeoCoord(xPix, yPix, inputRaster, sourceSR='', geomTransform='',
                    targetSR=''):
    """Convert pixel coordinates to geographic coordinates using rasterio."""
    with rasterio.open(inputRaster) as src:
        transform = src.transform
        crs = src.crs
    lon, lat = rasterio_xy(transform, yPix, xPix)  # rasterio uses (row, col) = (y, x)
    if targetSR != '':
        # reproject to target CRS if specified (targetSR expected as EPSG string or pyproj CRS)
        transformer = Transformer.from_crs(crs, targetSR, always_xy=True)
        lon, lat = transformer.transform(lon, lat)
    return (lon, lat)


###############################################################################
def nodes_near_point(x, y, kdtree, kd_idx_dic, x_coord='x', y_coord='y',
                     n_neighbors=-1,
                     radius_m=150,
                     verbose=False):
    """
    Get nodes near the given point.

    Notes
    -----
    if n_neighbors < 0, query based on distance,
    else just return n nearest neighbors

    Arguments
    ---------
    x : float
        x coordinate of point
    y: float
        y coordinate of point
    kdtree : scipy.spatial.kdtree
        kdtree of nondes in graph
    kd_idx_dic : dict
        Dictionary mapping kdtree entry to node name
    x_coord : str
        Name of x_coordinate, can be 'x' or 'lon'. Defaults to ``'x'``.
    y_coord : str
        Name of y_coordinate, can be 'y' or 'lat'. Defaults to ``'y'``.
    n_neighbors : int
        Neareast number of neighbors to return. If < 0, ignore.
        Defaults to ``-1``.
    radius_meters : float
        Radius to search for nearest neighbors
    Returns
    -------
    kd_idx_dic, kdtree, arr : tuple
        kd_idx_dic maps kdtree entry to node name
        kdree is the actual kdtree
        arr is the numpy array of node positions
    """

    point = [x, y]

    # query kd tree for nodes of interest
    if n_neighbors > 0:
        node_names, idxs_refine, dists_m_refine = _query_kd_nearest(
            kdtree, kd_idx_dic, point, n_neighbors=n_neighbors)
    else:
        node_names, idxs_refine, dists_m_refine = _query_kd_ball(
            kdtree, kd_idx_dic, point, radius_m)

    if verbose:
        print(("subgraph node_names:", node_names))

    return node_names, dists_m_refine


###############################################################################
def _nodes_near_origin(G_, node, kdtree, kd_idx_dic,
                       x_coord='x', y_coord='y', radius_m=150, verbose=False):
    """Get nodes a given radius from the desired node.

    G_ should be the maximally simplified graph.
    """

    # get node coordinates
    n_props = G_.nodes[node]
    x0, y0 = n_props[x_coord], n_props[y_coord]
    point = [x0, y0]

    # query kd tree for nodes of interest
    node_names, idxs_refine, dists_m_refine = _query_kd_ball(
        kdtree, kd_idx_dic, point, radius_m)
    if verbose:
        print(("subgraph node_names:", node_names))

    return node_names, dists_m_refine


###############################################################################
def G_to_kdtree(G_, x_coord='x', y_coord='y', verbose=False):
    """
    Create kd tree from node positions.

    Notes
    -----
    (x, y) = (lon, lat)
    kd_idx_dic maps kdtree entry to node name:
        kd_idx_dic[i] = n (n in G.nodes())
    x_coord can be in utm (meters), or longitude

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with nodes assumed to have a dictioary of
        properties that includes position
    x_coord : str
        Name of x_coordinate, can be 'x' or 'lon'. Defaults to ``'x'``.
    y_coord : str
        Name of y_coordinate, can be 'y' or 'lat'. Defaults to ``'y'``.

    Returns
    -------
    kd_idx_dic, kdtree, arr : tuple
        kd_idx_dic maps kdtree entry to node name
        kdree is the actual kdtree
        arr is the numpy array of node positions
    """

    nrows = len(G_.nodes())
    ncols = 2
    kd_idx_dic = {}
    arr = np.zeros((nrows, ncols))
    # populate node array
    t1 = time.time()
    for i, n in enumerate(G_.nodes()):
        n_props = G_.nodes[n]
        if x_coord == 'lon':
            lat, lon = n_props['lat'], n_props['lon']
            x, y = lon, lat
        else:
            x, y = n_props[x_coord], n_props[y_coord]

        arr[i] = [x, y]
        kd_idx_dic[i] = n

    # now create kdtree from numpy array
    kdtree = scipy.spatial.KDTree(arr)
    if verbose:
        print("Time to create k-d tree:", time.time() - t1, "seconds")
    return kd_idx_dic, kdtree, arr


###############################################################################
def _query_kd_nearest(kdtree, kd_idx_dic, point, n_neighbors=10,
                      distance_upper_bound=1000, keep_point=True):
    """
    Query the kd-tree for neighbors.

    Return nearest node names, distances, nearest node indexes.
    If not keep_point, remove the origin point from the list.
    """

    dists_m, idxs = kdtree.query(
        point, k=n_neighbors, distance_upper_bound=distance_upper_bound
    )

    idxs_refine = list(np.atleast_1d(np.asarray(idxs)))
    dists_m_refine = list(np.atleast_1d(np.asarray(dists_m)))
    node_names = [kd_idx_dic[i] for i in idxs_refine]

    return node_names, idxs_refine, dists_m_refine


###############################################################################
def _query_kd_ball(kdtree, kd_idx_dic, point, r_meters, keep_point=True):
    """
    Query the kd-tree for neighbors within a distance r of the point.

    Return nearest node names, distances, nearest node indexes.
    If not keep_point, remove the origin point from the list.
    """

    if r_meters == 0:
        dists_m, idxs = kdtree.query(point, k=1, distance_upper_bound=np.inf)
        dists_m = np.atleast_1d(np.asarray(dists_m))
        idxs = np.atleast_1d(np.asarray(idxs))
        f0 = np.where(dists_m == 0)
    else:
        dists_m, idxs = kdtree.query(point, k=500, distance_upper_bound=r_meters)
        dists_m = np.atleast_1d(np.asarray(dists_m))
        idxs = np.atleast_1d(np.asarray(idxs))
    # keep only points within distance and greater than 0?
    if r_meters != 0:
        if not keep_point:
            f0 = np.where((dists_m <= r_meters) & (dists_m > 0))
        else:
            f0 = np.where((dists_m <= r_meters))
    idxs_refine = list(idxs[f0])
    dists_m_refine = list(dists_m[f0])
    node_names = [kd_idx_dic[i] for i in idxs_refine]

    return node_names, idxs_refine, dists_m_refine


###############################################################################
def _get_graph_extent(G_):
    """Return min and max x and y of graph nodes."""
    xall = [G_.nodes[n]['x'] for n in G_.nodes()]
    yall = [G_.nodes[n]['y'] for n in G_.nodes()]
    xmin, xmax = np.min(xall), np.max(xall)
    ymin, ymax = np.min(yall), np.max(yall)
    dx, dy = xmax - xmin, ymax - ymin
    return xmin, xmax, ymin, ymax, dx, dy


###############################################################################
def _latlon2pixel(lat, lon, input_raster='', targetsr='', geom_transform=''):
    """Convert lat/lon to pixel coordinates using rasterio."""
    with rasterio.open(input_raster) as src:
        transform = src.transform
        crs = src.crs
    # transform lat/lon (EPSG:4326) to raster CRS
    transformer = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    x_proj, y_proj = transformer.transform(lon, lat)
    # convert projected coords to pixel
    row, col = rowcol(transform, x_proj, y_proj)
    return (col, row)  # (x_pix, y_pix)


###############################################################################
def _wmp2pixel(x, y, input_raster='', targetsr='', geom_transform=''):
    """Convert Web Mercator (EPSG:3857) coords to pixel coordinates."""
    with rasterio.open(input_raster) as src:
        transform = src.transform
        crs = src.crs
    transformer = Transformer.from_crs('EPSG:3857', crs, always_xy=True)
    x_proj, y_proj = transformer.transform(x, y)
    row, col = rowcol(transform, x_proj, y_proj)
    return (col, row)


###############################################################################
def _set_pix_coords(G_, im_test_file=''):
    """Get pixel coords.  Update G_ and get control_points, and graph_coords."""

    if len(G_.nodes()) == 0:
        return G_, [], []

    control_points, cp_x, cp_y = [], [], []
    for n in G_.nodes():
        u_x, u_y = G_.nodes[n]['x'], G_.nodes[n]['y']
        control_points.append([n, u_x, u_y])
        lat, lon = G_.nodes[n]['lat'], G_.nodes[n]['lon']
        if len(im_test_file) > 0:
            pix_x, pix_y = _latlon2pixel(lat, lon, input_raster=im_test_file)
        else:
            print("set_pix_coords(): oops, no image file")
            pix_x, pix_y = 0, 0
        # update G_
        G_.nodes[n]['pix_col'] = pix_x
        G_.nodes[n]['pix_row'] = pix_y
        G_.nodes[n]['x_pix'] = pix_x
        G_.nodes[n]['y_pix'] = pix_y
        # add to arrays
        cp_x.append(pix_x)
        cp_y.append(pix_y)
    # get line segments in pixel coords
    seg_endpoints = []
    for (u, v) in G_.edges():
        ux, uy = G_.nodes[u]['pix_col'], G_.nodes[u]['pix_row']
        vx, vy = G_.nodes[v]['pix_col'], G_.nodes[v]['pix_row']
        seg_endpoints.append([(ux, uy), (vx, vy)])
    gt_graph_coords = (cp_x, cp_y, seg_endpoints)

    return G_, control_points, gt_graph_coords


###############################################################################
def convertTo8Bit(rasterImageName, outputRaster,
                  outputPixType='Byte',
                  outputFormat='GTiff',
                  rescale_type='rescale',
                  percentiles=[2, 98]):
    """Convert raster to 8-bit using rasterio.

    Arguments
    ---------
    rasterImageName : str
        Path to input raster.
    outputRaster : str
        Path to output 8-bit raster.
    outputPixType : str
        Output pixel type. Defaults to ``'Byte'``.
    outputFormat : str
        Output format. Defaults to ``'GTiff'``.
    rescale_type : str
        One of 'rescale' or 'clip'. If 'rescale', each band is rescaled to
        its own percentile min/max. If 'clip', scaling is 0–65535.
        Defaults to ``'rescale'``.
    percentiles : list
        Lower and upper percentiles for rescaling. Defaults to ``[2, 98]``.
    """
    with rasterio.open(rasterImageName) as src:
        meta = src.meta.copy()
        data = src.read()

    nbands = data.shape[0]
    out_data = np.zeros_like(data, dtype=np.uint8)

    for band_idx in range(nbands):
        band_arr = data[band_idx].astype(float)
        if rescale_type == 'rescale':
            bmin = np.percentile(band_arr, percentiles[0])
            bmax = np.percentile(band_arr, percentiles[1])
        else:
            bmin, bmax = 0, 65535
        # clip and scale to 0-255
        band_arr = np.clip(band_arr, bmin, bmax)
        if bmax > bmin:
            band_arr = (band_arr - bmin) / (bmax - bmin) * 255
        out_data[band_idx] = band_arr.astype(np.uint8)

    meta.update({'dtype': 'uint8', 'driver': 'GTiff'})
    with rasterio.open(outputRaster, 'w', **meta) as dst:
        dst.write(out_data)


###############################################################################
def create_buffer_geopandas(inGDF, buffer_distance_meters=2,
                            buffer_cap_style=1, dissolve_by='class',
                            projectToUTM=True, verbose=False):
    """
    Create a buffer around the lines of the geojson.

    Arguments
    ---------
    inGDF : geodataframe
        Geodataframe from a SpaceNet geojson.
    buffer_distance_meters : float
        Width of buffer around geojson lines.  Formally, this is the distance
        to each geometric object.  Optional.  Defaults to ``2``.
    buffer_cap_style : int
        Cap_style of buffer, see: (https://shapely.readthedocs.io/en/stable/manual.html#constructive-methods)
        Defaults to ``1`` (round).
    dissolve_by : str
        Method for differentiating rows in geodataframe, and creating unique
        mask values.  Defaults to ``'class'``.
    projectToUTM : bool
        Switch to project gdf to UTM coordinates. Defaults to ``True``.
    verbose : bool
        Switch to print relevant values.  Defaults to ``False``.

    Returns
    -------
    gdf_buffer : geopandas dataframe
        Dataframe created from geojson
    """

    if len(inGDF) == 0:
        return []

    # Transform gdf Roadlines into UTM so that Buffer makes sense
    if projectToUTM:
        tmpGDF = osmnx_funcs.project_gdf(inGDF, inGDF.crs)
    else:
        tmpGDF = inGDF

    if verbose:
        print("inGDF.columns:", tmpGDF.columns)
    gdf_utm_buffer = tmpGDF.copy()

    # perform Buffer to produce polygons from Line Segments
    gdf_utm_buffer['geometry'] = tmpGDF.buffer(buffer_distance_meters,
                                               cap_style=buffer_cap_style)

    # dissolve preserves CRS in modern geopandas; no need to reassign
    gdf_utm_dissolve = gdf_utm_buffer.dissolve(by=dissolve_by)
    if projectToUTM:
        gdf_buffer = gdf_utm_dissolve.to_crs(inGDF.crs)
    else:
        gdf_buffer = gdf_utm_dissolve
    if verbose:
        print("gdf_buffer['geometry'].values[0]:",
              gdf_buffer['geometry'].values[0])

    # add the dissolve_by column back into final gdf, since it's now the index
    gdf_buffer[dissolve_by] = gdf_buffer.index.values

    return gdf_buffer


###############################################################################
def _get_road_buffer(geoJson, im_vis_file, output_raster,
                     buffer_meters=2, burnValue=1,
                     buffer_cap_style=6,
                     useSpacenetLabels=False,
                     plot_file='', figsize=(11, 3), fontsize=6,
                     dpi=800, show_plot=False,
                     valid_road_types=set([]), verbose=False):
    """
    Wrapper around create_buffer_geopandas(), with plots.

    Get buffer around roads defined by geojson and image files.
    valid_road_types serves as a filter of valid types (no filter if len==0).
    https://wiki.openstreetmap.org/wiki/Key:highway
    valid_road_types = set(['motorway', 'trunk', 'primary', 'secondary',
                            'tertiary',
                            'motorway_link', 'trunk_link', 'primary_link',
                            'secondary_link', 'tertiary_link',
                            'unclassified', 'residential', 'service'])
    """

    # filter out roads of the wrong type
    try:
        inGDF_raw = gpd.read_file(geoJson)
    except Exception:
        with rasterio.open(im_vis_file) as src:
            h, w = src.height, src.width
        mask_gray = np.zeros((h, w), dtype=np.uint8)
        with rasterio.open(output_raster, 'w', driver='GTiff',
                           height=h, width=w, count=1, dtype='uint8') as dst:
            dst.write(mask_gray[np.newaxis, ...])
        return [], []

    if useSpacenetLabels:
        inGDF = inGDF_raw
        try:
            inGDF['type'] = inGDF['road_type'].values
            inGDF['class'] = 'highway'
            inGDF['highway'] = 'highway'
        except Exception:
            pass

    else:
        # filter out roads of the wrong type
        if (len(valid_road_types) > 0) and (len(inGDF_raw) > 0):
            if 'highway' in inGDF_raw.columns:
                inGDF = inGDF_raw[inGDF_raw['highway'].isin(valid_road_types)]
                # set type tag
                inGDF['type'] = inGDF['highway'].values
                inGDF['class'] = 'highway'
            else:
                inGDF = inGDF_raw[inGDF_raw['type'].isin(valid_road_types)]
                # set highway tag
                inGDF['highway'] = inGDF['type'].values

            if verbose:
                print("gdf.type:", inGDF['type'])
                if len(inGDF) != len(inGDF_raw):
                    print("len(inGDF), len(inGDF_raw)",
                          len(inGDF), len(inGDF_raw))
                    print("gdf['type']:", inGDF['type'])
        else:
            inGDF = inGDF_raw
            try:
                inGDF['type'] = inGDF['highway'].values
                inGDF['class'] = 'highway'
            except Exception:
                pass

    gdf_buffer = create_buffer_geopandas(inGDF,
                                         buffer_distance_meters=buffer_meters,
                                         buffer_cap_style=buffer_cap_style,
                                         dissolve_by='class',
                                         projectToUTM=True)

    # make sure gdf is not null
    if len(gdf_buffer) == 0:
        with rasterio.open(im_vis_file) as src:
            h, w = src.height, src.width
        mask_gray = np.zeros((h, w), dtype=np.uint8)
        with rasterio.open(output_raster, 'w', driver='GTiff',
                           height=h, width=w, count=1, dtype='uint8') as dst:
            dst.write(mask_gray[np.newaxis, ...])
    # create label image
    else:
        gdf_to_array(gdf_buffer, im_vis_file, output_raster,
                     burnValue=burnValue)

    # load mask
    with rasterio.open(output_raster) as src:
        mask_gray = src.read(1)

    if plot_file:

        fig, (ax0, ax1, ax2, ax3) = plt.subplots(1, 4, figsize=figsize)

        # road lines
        try:
            gdfRoadLines = gpd.read_file(geoJson)
            gdfRoadLines.plot(ax=ax0, marker='o', color='red')
        except Exception:
            ax0.imshow(mask_gray)
        ax0.axis('off')
        ax0.set_aspect('equal')
        ax0.set_title('Unfiltered Roads from GeoJson', fontsize=fontsize)

        # first show raw image using rasterio
        with rasterio.open(im_vis_file) as src:
            # read up to 3 bands as RGB
            n_bands = src.count
            if n_bands >= 3:
                img_mpl = np.stack([src.read(1), src.read(2), src.read(3)],
                                   axis=-1)
            else:
                img_mpl = src.read(1)
        ax1.imshow(img_mpl)
        ax1.axis('off')
        ax1.set_title('Raw Image', fontsize=fontsize)

        # plot mask
        ax2.imshow(mask_gray)
        ax2.axis('off')
        ax2.set_title('Roads Mask (' + str(np.round(buffer_meters))
                      + ' meter buffer)', fontsize=fontsize)

        # plot combined
        ax3.imshow(img_mpl)
        # overlay mask
        # set zeros to nan
        z = mask_gray.astype(float)
        z[z == 0] = np.nan
        # change palette to orange
        palette = plt.cm.gray
        palette.set_over('orange', 1.0)
        ax3.imshow(z, cmap=palette, alpha=0.4,
                   norm=matplotlib.colors.Normalize(vmin=0.5, vmax=0.9,
                                                    clip=False))
        ax3.set_title('Raw Image + Buffered Roads', fontsize=fontsize)
        ax3.axis('off')

        plt.savefig(plot_file, dpi=dpi)
        if not show_plot:
            plt.close()

    return mask_gray, gdf_buffer


##############################################################################
def gdf_to_array(gdf, im_file, output_raster, burnValue=150,
                 mask_burn_val_key='', compress=True, NoData_value=0,
                 verbose=False):
    """
    Rasterize geodataframe to array matching im_file's grid, save as output_raster.

    Notes
    -----
    Uses rasterio.features.rasterize instead of GDAL RasterizeLayer.

    Arguments
    ---------
    gdf : geodataframe
        Input geodataframe with geometry column.
    im_file : str
        Path to image file corresponding to gdf.
    output_raster : str
        Output path of saved mask (should end in .tif).
    burnValue : int
        Value to burn to mask. Superceded by mask_burn_val_key.
        Defaults to ``150``.
    mask_burn_val_key : str
        Column name in gdf to use for mask burning. Supercedes burnValue.
        Defaults to ``''`` (in which case burnValue is used).
    compress : bool
        Switch to compress output raster. Defaults to ``True``.
    NoData_value : int
        Value to assign array if no data exists. If this value is <0
        (e.g. -9999), a null value will show in the image. Defaults to ``0``.
    verbose : bool
        Switch to print relevant values.  Defaults to ``False``.

    Returns
    -------
    None
    """

    with rasterio.open(im_file) as src:
        meta = src.meta.copy()
        transform = src.transform
        out_shape = (src.height, src.width)

    if verbose:
        print("transform:", transform)

    meta.update({'count': 1, 'dtype': 'uint8', 'driver': 'GTiff',
                 'nodata': NoData_value})
    if compress:
        meta['compress'] = 'lzw'

    # build shapes list for rasterize
    shapes = []
    for j, row in gdf.iterrows():
        geom = row['geometry']
        if len(mask_burn_val_key) > 0:
            burn_val = int(row[mask_burn_val_key])
            if verbose:
                print("burnVal:", burn_val)
        else:
            burn_val = burnValue
        shapes.append((geom, burn_val))

    if verbose:
        print("target meta:", meta)

    out_arr = rio_rasterize(shapes, out_shape=out_shape, transform=transform,
                            fill=NoData_value, dtype='uint8')

    with rasterio.open(output_raster, 'w', **meta) as dst:
        dst.write(out_arr, 1)


###############################################################################
def geojson_to_arr(image_path, geojson_path, mask_path_out_gray,
                   buffer_distance_meters=2, buffer_cap_style=1,
                   dissolve_by='speed_mph', mask_burn_val_key='burnValue',
                   min_burn_val=0, max_burn_val=255,
                   verbose=False):
    """
    Create buffer around geojson for desired geojson feature, save as mask.

    Arguments
    ---------
    image_path : str
        Path to input image corresponding to the geojson file.
    geojson_path : str
        Path to geojson file.
    mask_path_out_gray : str
        Output path of saved mask (should end in .tif).
    buffer_distance_meters : float
        Width of buffer around geojson lines.  Formally, this is the distance
        to each geometric object.  Optional.  Defaults to ``2``.
    buffer_cap_style : int
        Cap_style of buffer, see: (https://shapely.readthedocs.io/en/stable/manual.html#constructive-methods)
        Defaults to ``1`` (round).
    dissolve_by : str
        Method for differentiating rows in geodataframe, and creating unique
        mask values.  Defaults to ``'speed_m/s'``.
    mask_burn_val_key : str
        Column to name burn value in geodataframe. Defaults to ``'burnValue'``.
    min_burn_val : int
        Minimum value to burn to mask. Rescale all values linearly with this
        minimum value.  If <= 0, ignore.  Defaults to ``0``.
    max_burn_val : int
        Maximum value to burn to mask. Rescale all values linearly with this
        maxiumum value.  If <= 0, ignore.  Defaults to ``256``.
    verbose : bool
        Switch to print relevant values.  Defaults to ``False``.

    Returns
    -------
    gdf_buffer : geopandas dataframe
        Dataframe created from geojson
    """

    # get gdf_buffer
    try:
        inGDF = gpd.read_file(geojson_path)
    except TypeError:
        print("Empty mask for path:", geojson_path)
        # create empty mask using rasterio
        with rasterio.open(image_path) as src:
            h, w = src.height, src.width
        mask_gray = np.zeros((h, w), dtype=np.uint8)
        with rasterio.open(mask_path_out_gray, 'w', driver='GTiff',
                           height=h, width=w, count=1, dtype='uint8') as dst:
            dst.write(mask_gray[np.newaxis, ...])
        return []

    gdf_buffer = create_buffer_geopandas(
        inGDF, buffer_distance_meters=buffer_distance_meters,
        buffer_cap_style=buffer_cap_style, dissolve_by=dissolve_by,
        projectToUTM=False, verbose=verbose)

    if verbose:
        print("gdf_buffer.columns:", gdf_buffer.columns)
        print("gdf_buffer:", gdf_buffer)

    # set burn values
    burn_vals_raw = gdf_buffer[dissolve_by].values.astype(float)
    if verbose:
        print("burn_vals_raw:", burn_vals_raw)
    if (max_burn_val > 0) and (min_burn_val >= 0):
        scale_mult = (max_burn_val - min_burn_val) / np.max(burn_vals_raw)
        burn_vals = min_burn_val + scale_mult * burn_vals_raw
    else:
        burn_vals = burn_vals_raw
    if verbose:
        print("np.unique burn_vals:", np.sort(np.unique(burn_vals)))
    gdf_buffer[mask_burn_val_key] = burn_vals

    # create mask
    gdf_to_array(gdf_buffer, image_path, mask_path_out_gray,
                 mask_burn_val_key=mask_burn_val_key,
                 verbose=verbose)

    return gdf_buffer


###############################################################################
def _create_speed_arr(image_path, geojson_path, mask_path_out_gray,
                      bin_conversion_func, mask_burn_val_key='burnValue',
                      buffer_distance_meters=2, buffer_cap_style=1,
                      dissolve_by='speed_m/s', bin_conversion_key='speed_mph',
                      verbose=False):
    """
    Create buffer around geojson for speeds, use bin_conversion_func to
    assign values to the mask.

    Similar to geojson_to_arr().
    """

    # get gdf_buffer
    try:
        inGDF = gpd.read_file(geojson_path)
    except Exception:
        print("Empty mask for path:", geojson_path)
        # create empty mask using rasterio
        with rasterio.open(image_path) as src:
            h, w = src.height, src.width
        mask_gray = np.zeros((h, w), dtype=np.uint8)
        with rasterio.open(mask_path_out_gray, 'w', driver='GTiff',
                           height=h, width=w, count=1, dtype='uint8') as dst:
            dst.write(mask_gray[np.newaxis, ...])
        return []

    gdf_buffer = create_buffer_geopandas(
        inGDF, buffer_distance_meters=buffer_distance_meters,
        buffer_cap_style=buffer_cap_style, dissolve_by=dissolve_by,
        projectToUTM=True, verbose=verbose)

    # set burn values
    speed_arr = gdf_buffer[bin_conversion_key].values
    burnVals = [bin_conversion_func(s) for s in speed_arr]
    gdf_buffer[mask_burn_val_key] = burnVals

    # create mask
    gdf_to_array(gdf_buffer, image_path, mask_path_out_gray,
                 mask_burn_val_key=mask_burn_val_key, verbose=verbose)

    return gdf_buffer


###############################################################################
def create_speed_gdf_v0(image_path, geojson_path, mask_path_out_gray,
                        bin_conversion_func, mask_burn_val_key='burnValue',
                        buffer_distance_meters=2, buffer_cap_style=1,
                        dissolve_by='speed_m/s', bin_conversion_key='speed_mph',
                        verbose=False):
    """
    Create buffer around geojson for speeds, use bin_conversion_func to
    assign values to the mask.
    """

    # get gdf_buffer
    try:
        inGDF = gpd.read_file(geojson_path)
    except Exception:
        print("Empty mask for path:", geojson_path)
        # create empty mask using rasterio
        with rasterio.open(image_path) as src:
            h, w = src.height, src.width
        mask_gray = np.zeros((h, w), dtype=np.uint8)
        with rasterio.open(mask_path_out_gray, 'w', driver='GTiff',
                           height=h, width=w, count=1, dtype='uint8') as dst:
            dst.write(mask_gray[np.newaxis, ...])
        return []

    # project
    projGDF = osmnx_funcs.project_gdf(inGDF)
    if verbose:
        print("inGDF.columns:", inGDF.columns)

    gdf_utm_buffer = projGDF.copy()
    # perform Buffer to produce polygons from Line Segments
    gdf_utm_buffer['geometry'] = gdf_utm_buffer.buffer(buffer_distance_meters,
                                                       buffer_cap_style)
    # dissolve preserves CRS in modern geopandas; no need to reassign
    gdf_utm_dissolve = gdf_utm_buffer.dissolve(by=dissolve_by)
    gdf_buffer = gdf_utm_dissolve.to_crs(inGDF.crs)
    if verbose:
        print("gdf_buffer['geometry'].values[0]:",
              gdf_buffer['geometry'].values[0])

    # set burn values
    speed_arr = gdf_buffer[bin_conversion_key].values
    burnVals = [bin_conversion_func(s) for s in speed_arr]
    gdf_buffer[mask_burn_val_key] = burnVals

    # create mask
    gdf_to_array(gdf_buffer, image_path, mask_path_out_gray,
                 mask_burn_val_key=mask_burn_val_key, verbose=verbose)

    return gdf_buffer


###############################################################################
def convert_array_to_multichannel(in_arr, n_channels=7, burnValue=255,
                                  append_total_band=False, verbose=False):
    """
    Take input array with multiple values, make each value a unique channel.

    Assume a zero value is background, while value of 1 is the first channel,
    2 the second channel, etc.
    """

    h, w = in_arr.shape[:2]
    out_arr = np.zeros((n_channels, h, w), dtype=np.uint8)

    for band in range(n_channels):
        val = band + 1
        band_out = np.zeros((h, w), dtype=np.uint8)
        if verbose:
            print("band:", band)
        band_arr_bool = np.where(in_arr == val)
        band_out[band_arr_bool] = burnValue
        out_arr[band, :, :] = band_out

    if append_total_band:
        tot_band = np.zeros((h, w), dtype=np.uint8)
        band_arr_bool = np.where(in_arr > 0)
        tot_band[band_arr_bool] = burnValue
        tot_band = tot_band.reshape(1, h, w)
        out_arr = np.concatenate((out_arr, tot_band), axis=0).astype(np.uint8)

    if verbose:
        print("out_arr.shape:", out_arr.shape)
    return out_arr


### Helper Functions
###############################################################################
def CreateMultiBandGeoTiff(OutPath, Array):
    """Write multi-band array to GeoTiff. Array shape: (Channels, Y, X)."""
    n_bands, height, width = Array.shape
    meta = {'driver': 'GTiff', 'dtype': 'uint8', 'width': width,
            'height': height, 'count': n_bands, 'compress': 'lzw'}
    with rasterio.open(OutPath, 'w', **meta) as dst:
        for i in range(n_bands):
            dst.write(Array[i], i + 1)
    return OutPath


###############################################################################
def geomGeo2geomPixel(geom, affineObject=None, input_raster='',
                      gdal_geomTransform=[]):
    """
    Transform a shapely geometry from geospatial to pixel coordinates.

    Based on spacenet utilities v3 geotools.py.

    Arguments
    ---------
    geom : shapely geometry
        Input geometry in geospatial coordinates.
    affineObject : Affine or None
        Affine transform object. If None or empty, derived from input_raster
        or gdal_geomTransform. Defaults to ``None``.
    input_raster : str
        Path to raster file to read transform from. Defaults to ``''``.
    gdal_geomTransform : list
        GDAL-style geotransform coefficients. Defaults to ``[]``.

    Returns
    -------
    shapely geometry in pixel coordinates
    """
    if affineObject is None or affineObject == []:
        if input_raster != '':
            with rasterio.open(input_raster) as src:
                affineObject = src.transform
        elif gdal_geomTransform != []:
            affineObject = Affine.from_gdal(*gdal_geomTransform)
        else:
            return geom

    affineObjectInv = ~affineObject

    geomTransform = shapely.affinity.affine_transform(
        geom,
        [affineObjectInv.a,
         affineObjectInv.b,
         affineObjectInv.d,
         affineObjectInv.e,
         affineObjectInv.xoff,
         affineObjectInv.yoff]
    )

    return geomTransform


###############################################################################
def geomPixel2geomGeo(geom, affineObject=None, input_raster='',
                      gdal_geomTransform=[]):
    """
    Transform a shapely geometry from pixel to geospatial coordinates.

    Based on spacenet utilities v3 geotools.py.

    Arguments
    ---------
    geom : shapely geometry
        Input geometry in pixel coordinates.
    affineObject : Affine or None
        Affine transform object. If None or empty, derived from input_raster
        or gdal_geomTransform. Defaults to ``None``.
    input_raster : str
        Path to raster file to read transform from. Defaults to ``''``.
    gdal_geomTransform : list
        GDAL-style geotransform coefficients. Defaults to ``[]``.

    Returns
    -------
    shapely geometry in geospatial coordinates
    """
    if affineObject is None or affineObject == []:
        if input_raster != '':
            with rasterio.open(input_raster) as src:
                affineObject = src.transform
        elif gdal_geomTransform != []:
            affineObject = Affine.from_gdal(*gdal_geomTransform)
        else:
            return geom

    geomTransform = shapely.affinity.affine_transform(
        geom,
        [affineObject.a,
         affineObject.b,
         affineObject.d,
         affineObject.e,
         affineObject.xoff,
         affineObject.yoff]
    )

    return geomTransform


###############################################################################
def _haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points in metres.

    Points are on the earth, specified in decimal degrees.
    http://stackoverflow.com/questions/15736995/how-can-i-
        quickly-estimate-the-distance-between-two-latitude-longitude-points
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = list(map(radians, [lon1, lat1, lon2, lat2]))
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    m = 1000. * km
    return m


###############################################################################
def get_gsd(im_test_file):
    """Return GSD in meters."""
    with rasterio.open(im_test_file) as src:
        transform = src.transform
    ulX = transform.c   # upper-left x (lon)
    ulY = transform.f   # upper-left y (lat)
    yDist = transform.e  # pixel height (negative)
    dy = _haversine(ulX, ulY, ulX, ulY + yDist)
    return dy


###############################################################################
def get_extent(srcFileImage):
    """Get spatial extent of raster."""
    with rasterio.open(srcFileImage) as src:
        bounds = src.bounds
        transform = src.transform
        xsize = src.width
        ysize = src.height
    xres = transform.a
    yres = transform.e
    xmin = bounds.left + xres * 0.5
    xmax = bounds.left + (xres * xsize) - xres * 0.5
    ymin = bounds.top + (yres * ysize) + yres * 0.5
    ymax = bounds.top - yres * 0.5
    return xmin, ymin, xmax, ymax


###############################################################################
def get_pixel_dist_from_meters(im_test_file, len_meters):
    """
    Determine pixel distance for a given length in metres.

    For the input image, we want a buffer or other distance in meters;
    this function determines the pixel distance by calculating the GSD.
    """
    gsd = get_gsd(im_test_file)
    pix_width = max(1, np.rint(len_meters / gsd))

    return gsd, pix_width


###############################################################################
def get_unique(seq, idfun=None):
    """Return unique values from seq preserving order.

    https://www.peterbe.com/plog/uniqifiers-benchmark
    """
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        if marker in seen:
            continue
        seen[marker] = 1
        result.append(item)
    return result


###############################################################################
def _get_node_positions(G_, x_coord='x', y_coord='y'):
    """Get position array for all nodes."""
    nrows = len(G_.nodes())
    ncols = 2
    arr = np.zeros((nrows, ncols))
    # populate node array
    for i, n in enumerate(G_.nodes()):
        n_props = G_.nodes[n]
        x, y = n_props[x_coord], n_props[y_coord]
        arr[i] = [x, y]
    return arr
