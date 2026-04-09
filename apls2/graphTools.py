#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 13:30:14 2017

@author: avanetten

Heavily modified from spacenet utilities graphtools.
Modernized: replaced fiona with geopandas, removed sys.path hacks,
updated deprecated pandas/geopandas APIs, updated CRS strings.
"""

import shapely.geometry
import numpy as np
import os
import time
import pickle
import json
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from . import osmnx_funcs


###############################################################################
def parse_OGR_nodes_paths(vectorFileName, osmidx=0, osmNodeidx=0,
                          nodeListGpd=None,
                          valid_road_types=set([]),
                          roadTypeField='type',
                          verbose=True,
                          super_verbose=False):
    """
    Construct dicts of nodes and paths with key=osmid and value=dict of
    attributes.

    Notes
    -----
    valid_road_types is a set of road types to be allowed.

    Parameters
    ----------
    vectorFileName : str
        Absolute path to a vector file readable by geopandas (GeoJSON,
        Shapefile, GPKG, etc.) containing line segments.
    osmidx : int
        Starting edge index.
    osmNodeidx : int
        Starting node index.
    nodeListGpd : GeoDataFrame
        Accumulator of already-seen nodes.
    valid_road_types : set
        Set of road type strings to allow; empty set means allow all.
    roadTypeField : str
        Property field name that contains road type.
    verbose : bool
        Print progress information.
    super_verbose : bool
        Print detailed per-feature information.

    Returns
    -------
    nodes, paths : tuple of dicts
    """

    def _load_gdf_or_geojson(path):
        """Load vector data, falling back to direct GeoJSON parsing if needed."""
        try:
            return gpd.read_file(path)
        except Exception:
            if not str(path).lower().endswith((".geojson", ".json")):
                raise

        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)

        features = payload.get("features", [])
        records = []
        for feature in features:
            geom = feature.get("geometry")
            if geom is None:
                continue
            record = dict(feature.get("properties") or {})
            record["geometry"] = shapely.geometry.shape(geom)
            records.append(record)

        return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")

    if nodeListGpd is None or "geometry" not in nodeListGpd.columns:
        nodeListGpd = gpd.GeoDataFrame(
            {"geometry": [], "osmid": [], "x": [], "y": []},
            geometry="geometry",
            crs="EPSG:4326",
        )

    # ensure valid vectorFileName
    try:
        gdf = _load_gdf_or_geojson(vectorFileName)
        doit = True
    except Exception:
        doit = False
        return {}, {}

    if doit:
        nodes = {}
        paths = {}
        for i, row in gdf.iterrows():

            geom = row.geometry
            properties = {col: row[col] for col in gdf.columns if col != 'geometry'}

            # determine road type from available property fields
            if roadTypeField in properties:
                road_type = properties['type']
            elif 'highway' in properties:
                road_type = properties['highway']
            elif 'road_type' in properties:
                road_type = properties['road_type']
            else:
                road_type = 'None'

            if ((i % 100) == 0) and verbose:
                print("\n", i, "/", len(gdf))
                print("   geom:", geom)
                print("   properties:", properties)
                print("   road_type:", road_type)

            # check if road type allowable, continue if not
            if (len(valid_road_types) > 0) and \
                    (geom.geom_type == 'LineString'
                     or geom.geom_type == 'MultiLineString'):
                if road_type not in valid_road_types:
                    if verbose:
                        print("Invalid road type, skipping...")
                    continue

            # skip empty linestrings
            if 'LINESTRING EMPTY' in list(properties.values()):
                continue

            osmidx = int(osmidx + 1)

            if geom.geom_type == 'LineString':
                lineString = geom
                if super_verbose:
                    print("lineString.wkt:", lineString.wkt)

                path, nodeList, osmNodeidx, nodeListGpd = \
                    processLineStringFeature(lineString, osmidx, osmNodeidx,
                                             nodeListGpd, properties=properties)
                osmNodeidx = osmNodeidx + 1
                osmidx = osmidx + 1
                nodes.update(nodeList)
                paths[osmidx] = path

            elif geom.geom_type == 'MultiLineString':
                for linestring in geom.geoms:

                    path, nodeList, osmNodeidx, nodeListGpd = \
                        processLineStringFeature(linestring, osmidx, osmNodeidx,
                                                 nodeListGpd, properties=properties)
                    osmNodeidx = osmNodeidx + 1
                    osmidx = osmidx + 1
                    nodes.update(nodeList)
                    paths[osmidx] = path

    return nodes, paths


###############################################################################
def processLineStringFeature(lineString, keyEdge, osmNodeidx,
                             nodeListGpd=None, properties={}, 
                             roadTypeField='type'):
    """
    Iterate over points in a LineString and build node/path dicts.

    Parameters
    ----------
    lineString : shapely.geometry.LineString
        The line geometry to process.
    keyEdge : int
        The osmid to assign to this edge.
    osmNodeidx : int
        Current running node index.
    nodeListGpd : GeoDataFrame
        Accumulator of already-seen nodes.
    properties : dict
        Feature properties to attach to nodes and edges.
    roadTypeField : str
        Property field name that contains road type.

    Returns
    -------
    path, nodes, osmNodeidx, nodeListGpd : tuple
    """

    if nodeListGpd is None or "geometry" not in nodeListGpd.columns:
        nodeListGpd = gpd.GeoDataFrame(
            {"geometry": [], "osmid": [], "x": [], "y": []},
            geometry="geometry",
            crs="EPSG:4326",
        )

    osmNodeidx = osmNodeidx + 1
    path = {}
    nodes = {}
    path['osmid'] = keyEdge

    nodeList = []

    for point in lineString.coords:

        pointShp = shapely.geometry.shape(Point(point))
        if nodeListGpd.size == 0:
            nodeId = np.array([], dtype=int)
        else:
            nodeId = nodeListGpd[
                (nodeListGpd["x"] == point[0]) & (nodeListGpd["y"] == point[1])
            ]['osmid'].values

        if nodeId.size == 0:
            nodeId = osmNodeidx
            new_node = gpd.GeoDataFrame(
                [{"geometry": pointShp, "osmid": osmNodeidx,
                  "x": point[0], "y": point[1]}],
                geometry="geometry",
                crs="EPSG:4326",
            )
            nodeListGpd = gpd.GeoDataFrame(
                pd.concat([nodeListGpd, new_node], ignore_index=True),
                geometry="geometry",
                crs=new_node.crs,
            )
            osmNodeidx = osmNodeidx + 1

            node = {}
            # add coordinates
            node['x'] = point[0]
            node['y'] = point[1]
            node['osmid'] = nodeId

            # add properties
            for key, value in list(properties.items()):
                node[key] = value
            if roadTypeField in properties:
                node['highway'] = properties['type']
            else:
                node['highway'] = 'unclassified'

            nodes[nodeId] = node

        else:
            nodeId = nodeId[0]

        nodeList.append(nodeId)

    path['nodes'] = nodeList
    # add properties
    for key, value in list(properties.items()):
        path[key] = value
    # also set 'highway' flag
    if roadTypeField in properties:
        path['highway'] = properties['type']
    else:
        path['highway'] = 'unclassified'

    return path, nodes, osmNodeidx, nodeListGpd


###############################################################################
def create_graphGeoJson(geoJson, name='unnamed', retain_all=True,
                        network_type='all_private', valid_road_types=set([]),
                        roadTypeField='type',
                        osmidx=0, osmNodeidx=0,
                        verbose=True, super_verbose=False):
    """
    Create a networkx graph from a GeoJSON (or any geopandas-readable) file.

    Parameters
    ----------
    geoJson : str
        Path to a vector file supported by geopandas (GeoJSON, Shapefile,
        GPKG, etc.) containing line segments.
    name : str
        The name of the graph.
    retain_all : bool
        If True, return the entire graph even if it is not connected.
    network_type : str
        What type of network to create.
    valid_road_types : set
        Set of allowable road type strings; empty set means allow all.
    roadTypeField : str
        Property field name that contains road type.
    osmidx : int
        Starting edge index.
    osmNodeidx : int
        Starting node index.
    verbose : bool
        Print progress information.
    super_verbose : bool
        Print detailed per-feature information.

    Returns
    -------
    G : networkx MultiDiGraph
    """

    print('Creating networkx graph from downloaded OSM data...')
    start_time = time.time()

    # create the graph as a MultiDiGraph and set the original CRS to EPSG:4326
    G = nx.MultiDiGraph(name=name, crs='EPSG:4326')

    # extract nodes and paths from the downloaded osm data
    nodes = {}
    paths = {}

    if verbose:
        print("Running parse_OGR_nodes_paths...")
    nodes_temp, paths_temp = parse_OGR_nodes_paths(
        geoJson,
        valid_road_types=valid_road_types,
        verbose=verbose,
        super_verbose=super_verbose,
        osmidx=osmidx,
        osmNodeidx=osmNodeidx,
        roadTypeField=roadTypeField
    )

    if len(nodes_temp) == 0:
        return G
    if verbose:
        print(f"len(nodes_temp): {len(nodes_temp)}")
        print(f"len(paths_temp): {len(paths_temp)}")

    # add node props
    for key, value in list(nodes_temp.items()):
        nodes[key] = value
        if super_verbose:
            print(f"node key: {key}")
            print(f"  node value: {value}")

    # add edge props
    for key, value in list(paths_temp.items()):
        paths[key] = value
        if super_verbose:
            print(f"path key: {key}")
            print(f"  path value: {value}")

    # add each node to the graph
    for node, data in list(nodes.items()):
        G.add_node(node, **data)

    # add each way (aka, path) to the graph
    if super_verbose:
        print(f"paths: {paths}")
    G = osmnx_funcs.add_paths(G, paths, network_type)

    # retain only the largest connected component, if caller did not set
    # retain_all=True
    if not retain_all:
        G = osmnx_funcs.get_largest_component(G)

    # add length (great circle distance between nodes) attribute to each edge
    # to use as weight
    G = osmnx_funcs.add_edge_lengths(G)

    print('Created graph with {:,} nodes and {:,} edges in {:,.2f} seconds'.format(
        len(list(G.nodes())), len(list(G.edges())), time.time() - start_time
    ))

    return G


###############################################################################
if __name__ == "__main__":

    # test
    truth_dir = '/raid/cosmiq/spacenet/data/spacenetv2/spacenetLabels/AOI_2_Vegas/400m/'
    out_pkl = (
        '/raid/cosmiq/spacenet/data/spacenetv2/spacenetLabels/'
        'AOI_2_Vegas/spacenetroads_AOI_2_Vegas_img10_graphTools.pkl'
    )

    geoJson = os.path.join(
        truth_dir, 'spacenetroads_AOI_2_Vegas_img10.geojson'
    )
    G0 = create_graphGeoJson(geoJson, name='unnamed',
                             retain_all=True,
                             verbose=True)
    # pkl
    with open(out_pkl, 'wb') as f:
        pickle.dump(G0, f)
