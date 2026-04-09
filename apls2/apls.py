#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
APLS (Average Path Length Similarity) metric engine.

Modernized from the original CosmiQ Works apls package:
  - Replaced utm with pyproj
  - Replaced nx.read_gpickle / nx.write_gpickle with pickle
  - Replaced nx.connected_component_subgraphs with modern nx.connected_components
  - Removed sys.path hacks; uses relative package imports
  - Compatible with networkx >= 3.0, pandas >= 2.0, shapely >= 2.0
"""

import pickle
import networkx as nx
import scipy.spatial
import scipy.stats
import numpy as np
import random
import copy
import argparse
import os
import time
import math
import matplotlib.pyplot as plt
import pandas as pd
import shapely.wkt
import shapely.ops
from shapely.geometry import Point, LineString

from . import apls_utils
from . import apls_plots
from . import osmnx_funcs
from . import graphTools
from . import wkt_to_G
from . import topo_metric
from . import sp_metric


###############################################################################
def _read_graph(path):
    """Load a pickled NetworkX graph from path."""
    with open(path, 'rb') as f:
        return pickle.load(f)


###############################################################################
def _write_graph(G, path):
    """Pickle a NetworkX graph to path."""
    with open(path, 'wb') as f:
        pickle.dump(G, f)


###############################################################################
def _ensure_edge_speed(G_, speed_key='inferred_speed_mps', default_speed=13.41):
    """Backfill missing edge speed attributes on bundled legacy graphs."""
    for _, _, data in G_.edges(data=True):
        if speed_key in data:
            continue
        if 'speed_m/s' in data:
            data[speed_key] = data['speed_m/s']
        else:
            data[speed_key] = default_speed
    return G_


###############################################################################
def add_travel_time(G_, speed_key='inferred_speed_mps', length_key='length',
                    travel_time_key='travel_time_s', default_speed=13.41,
                    verbose=False):
    """
    Compute and add travel time estimates to each graph edge.

    Arguments
    ---------
    G_ : networkx graph
        Input graph; edges must have speed and length attributes.
    speed_key : str
        Edge attribute key for speed (m/s). Defaults to ``'inferred_speed_mps'``.
    length_key : str
        Edge attribute key for length (m). Defaults to ``'length'``.
    travel_time_key : str
        Name for the new travel time attribute. Defaults to ``'travel_time_s'``.
    default_speed : float
        Fallback speed in m/s when speed_key is missing. Defaults to ``13.41``.
    verbose : bool
        Print debug info. Defaults to ``False``.

    Returns
    -------
    G_ : networkx graph
        Updated graph with travel time on each edge.
    """
    for u, v, data in G_.edges(data=True):
        if speed_key in data:
            speed = data[speed_key]
            if isinstance(speed, list):
                speed = np.mean(speed)
        else:
            print("speed_key not found:", speed_key)
            return None
        if verbose:
            print("data[length_key]:", data[length_key], "speed:", speed)
        data[travel_time_key] = data[length_key] / speed
    return G_


###############################################################################
def create_edge_linestrings(G_, remove_redundant=True, verbose=False):
    """
    Ensure all edges have a Shapely LineString in their 'geometry' attribute.

    Arguments
    ---------
    G_ : networkx graph
    remove_redundant : bool
        Remove duplicate edges. Defaults to ``True``.
    verbose : bool

    Returns
    -------
    G_ : networkx graph
    """
    edge_seen_set = set()
    geom_seen = []
    bad_edges = []

    for i, (u, v, data) in enumerate(G_.edges(data=True)):
        if 'geometry' not in data:
            sourcex, sourcey = G_.nodes[u]['x'], G_.nodes[u]['y']
            targetx, targety = G_.nodes[v]['x'], G_.nodes[v]['y']
            line_geom = LineString([Point(sourcex, sourcey),
                                    Point(targetx, targety)])
            data['geometry'] = line_geom
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
        else:
            line_geom = data['geometry']
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
            if dist_to_u > dist_to_v:
                data['geometry'] = line_geom_rev

        if remove_redundant:
            if i == 0:
                edge_seen_set = {(u, v), (v, u)}
                geom_seen.append(line_geom)
            else:
                if (u, v) in edge_seen_set or (v, u) in edge_seen_set:
                    for geom_seen_tmp in geom_seen:
                        if (line_geom == geom_seen_tmp) \
                                or (line_geom_rev == geom_seen_tmp):
                            bad_edges.append((u, v))
                            if verbose:
                                print("Redundant edge:", u, v)
                else:
                    edge_seen_set.add((u, v))
                    geom_seen.append(line_geom)
                    geom_seen.append(line_geom_rev)

    if remove_redundant:
        if verbose:
            print("redundant edges:", bad_edges)
        for (u, v) in bad_edges:
            if G_.has_edge(u, v):
                G_.remove_edge(u, v)

    return G_


###############################################################################
def cut_linestring(line, distance, verbose=False):
    """
    Cut a Shapely LineString at ``distance`` meters from its start.

    Returns the original line unchanged if distance <= 0 or >= line.length.

    Arguments
    ---------
    line : shapely LineString
    distance : float
    verbose : bool

    Returns
    -------
    list of LineString segments (1 or 2 elements).
    """
    if verbose:
        print("Cutting linestring at distance", distance, "...")
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]

    coords = list(line.coords)
    for i, p in enumerate(coords):
        pdl = line.project(Point(p))
        if verbose:
            print(i, p, "line.project point:", pdl)
        if pdl == distance:
            return [LineString(coords[:i+1]), LineString(coords[i:])]
        if pdl > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:]),
            ]

    # self-loop case: interpolated point is past the last midpoint
    i = len(coords) - 1
    cp = line.interpolate(distance)
    return [
        LineString(coords[:i] + [(cp.x, cp.y)]),
        LineString([(cp.x, cp.y)] + coords[i:]),
    ]


###############################################################################
def get_closest_edge_from_G(G_, point, nearby_nodes_set=None, verbose=False):
    """
    Return the closest edge to ``point`` and the distance to it.

    Arguments
    ---------
    G_ : networkx graph
    point : shapely Point
    nearby_nodes_set : set or None
        If non-empty, restrict search to edges with a node in this set.
    verbose : bool

    Returns
    -------
    best_edge, min_dist, best_geom : tuple
    """
    if nearby_nodes_set is None:
        nearby_nodes_set = set()

    dist_list = []
    edge_list = []
    geom_list = []
    p = point

    for u, v, key, data in G_.edges(keys=True, data=True):
        if nearby_nodes_set and (u not in nearby_nodes_set) \
                and (v not in nearby_nodes_set):
            continue
        if verbose:
            print("u,v,key,data:", u, v, key, data)
        try:
            line = data['geometry']
        except KeyError:
            line = data['attr_dict']['geometry']
        geom_list.append(line)
        dist_list.append(p.distance(line))
        edge_list.append([u, v, key])

    min_idx = np.argmin(dist_list)
    return edge_list[min_idx], dist_list[min_idx], geom_list[min_idx]


###############################################################################
def insert_point_into_G(G_, point, node_id=100000, max_distance_meters=5,
                        nearby_nodes_set=None, allow_renaming=True,
                        verbose=False, super_verbose=False):
    """
    Insert a new node into the graph at the location nearest to ``point``.

    Only inserts if the nearest edge is within ``max_distance_meters``.
    Splits the nearest edge into two edges at the snap point.

    Arguments
    ---------
    G_ : networkx graph
    point : shapely Point
    node_id : int
        Unique ID for the new node. Defaults to ``100000``.
    max_distance_meters : float
    nearby_nodes_set : set or None
    allow_renaming : bool
        Relabel an existing node when the snap point is coincident with it.
    verbose : bool
    super_verbose : bool

    Returns
    -------
    G_, node_props, x, y : tuple
        Returns ``(G_, {}, -1, -1)`` when the point is too far or node_id
        already exists.
    """
    if nearby_nodes_set is None:
        nearby_nodes_set = set()

    best_edge, min_dist, best_geom = get_closest_edge_from_G(
        G_, point, nearby_nodes_set=nearby_nodes_set, verbose=super_verbose)
    [u, v, key] = best_edge
    G_node_set = set(G_.nodes())

    if verbose:
        print("Inserting point:", node_id)
        print("best edge:", best_edge, "dist:", min_dist)

    if min_dist > max_distance_meters:
        if verbose:
            print("min_dist > max_distance_meters, skipping...")
        return G_, {}, -1, -1

    if node_id in G_node_set:
        if verbose:
            print("Node ID:", node_id, "already exists, skipping...")
        return G_, {}, -1, -1

    line_geom = best_geom
    line_proj = line_geom.project(point)
    new_point = line_geom.interpolate(line_proj)
    x, y = new_point.x, new_point.y

    # build lat/lon metadata for the new node (best-effort)
    try:
        from pyproj import Transformer, CRS
        node_u = G_.nodes[u]
        if 'lat' in node_u and 'lon' in node_u:
            # graph is in UTM; back-project the new point to WGS84
            utm_crs = CRS.from_user_input(
                G_.graph.get('crs', 'EPSG:32601'))
            transformer = Transformer.from_crs(utm_crs, 'EPSG:4326',
                                               always_xy=True)
            lon, lat = transformer.transform(x, y)
        else:
            lat, lon = y, x
    except Exception:
        lat, lon = y, x

    node_props = {'highway': 'insertQ', 'lat': lat, 'lon': lon,
                  'osmid': node_id, 'x': x, 'y': y}
    G_.add_node(node_id, **node_props)

    _, _, edge_props_new = copy.deepcopy(
        list(G_.edges([u, v], data=True))[0])

    split_line = cut_linestring(line_geom, line_proj)
    if split_line is None:
        print("Failure in cut_linestring()...")
        return G_, {}, 0, 0

    if verbose:
        print("split_line:", split_line)

    if len(split_line) == 1:
        # new point coincides with an existing node
        outnode = ''
        outnode_x, outnode_y = -1, -1
        x_p, y_p = new_point.x, new_point.y
        x_u, y_u = G_.nodes[u]['x'], G_.nodes[u]['y']
        x_v, y_v = G_.nodes[v]['x'], G_.nodes[v]['y']
        buff = 0.05  # metres
        if abs(x_p - x_u) <= buff and abs(y_p - y_u) <= buff:
            outnode, outnode_x, outnode_y = u, x_u, y_u
        elif abs(x_p - x_v) <= buff and abs(y_p - y_v) <= buff:
            outnode, outnode_x, outnode_y = v, x_v, y_v
        else:
            print("Error: cannot determine node coincident with", node_id)
            return G_, {}, 0, 0

        if allow_renaming:
            node_props = G_.nodes[outnode]
            Gout = nx.relabel_nodes(G_, {outnode: node_id})
            if verbose:
                print("Swapping out node ids:", {outnode: node_id})
            return Gout, node_props, x_p, y_p
        else:
            line1 = LineString([new_point, Point(outnode_x, outnode_y)])
            edge_props_line1 = edge_props_new.copy()
            edge_props_line1['length'] = line1.length
            edge_props_line1['geometry'] = line1
            if line1.length > buff:
                print("Nodes should be coincident and length 0!")
                return G_, {}, 0, 0
            G_.add_edge(node_id, outnode, **edge_props_line1)
            return G_, node_props, x, y

    else:
        line1, line2 = split_line
        u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
        v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
        geom_p0 = list(line_geom.coords)[0]
        dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
        dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
        if dist_to_v < dist_to_u:
            line2, line1 = split_line

        if verbose:
            print("Creating two edges from split...")
            print("  original_length:", line_geom.length)
            print("  line1_length:", line1.length)
            print("  line2_length:", line2.length)

        edge_props_line1 = edge_props_new.copy()
        edge_props_line1['length'] = line1.length
        edge_props_line1['geometry'] = line1

        edge_props_line2 = edge_props_new.copy()
        edge_props_line2['length'] = line2.length
        edge_props_line2['geometry'] = line2

        geom_p0 = list(line_geom.coords)[0]
        dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
        dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
        if dist_to_u < dist_to_v:
            G_.add_edge(u, node_id, **edge_props_line1)
            G_.add_edge(node_id, v, **edge_props_line2)
        else:
            G_.add_edge(node_id, u, **edge_props_line1)
            G_.add_edge(v, node_id, **edge_props_line2)

        if verbose:
            print("insert edges:", u, '-', node_id, 'and', node_id, '-', v)

        G_.remove_edge(u, v, key)
        return G_, node_props, x, y


###############################################################################
def insert_control_points(G_, control_points, max_distance_meters=10,
                          allow_renaming=True, n_nodes_for_kd=1000,
                          n_neighbors=20, x_coord='x', y_coord='y',
                          verbose=True, super_verbose=False):
    """
    Insert a list of control points into graph G_.

    Arguments
    ---------
    G_ : networkx graph
    control_points : list of [node_id, x, y]
    max_distance_meters : float
    allow_renaming : bool
    n_nodes_for_kd : int
        Build a KD-tree when the graph exceeds this size.
    n_neighbors : int
        Neighbours to fetch from the KD-tree. Defaults to ``20``.
    x_coord, y_coord : str
    verbose, super_verbose : bool

    Returns
    -------
    Gout, new_xs, new_ys : tuple
    """
    t0 = time.time()

    if len(G_.nodes()) > n_nodes_for_kd:
        kd_idx_dic, kdtree, _ = apls_utils.G_to_kdtree(G_)

    Gout = G_.copy()
    new_xs, new_ys = [], []
    if len(G_.nodes()) == 0:
        return Gout, new_xs, new_ys

    for i, (node_id, x, y) in enumerate(control_points):
        if math.isinf(x) or math.isinf(y):
            print("Infinity in coords!:", x, y)
            return Gout, new_xs, new_ys

        if (i % 20) == 0:
            print(i, "/", len(control_points),
                  "Insert control point:", node_id, "x =", x, "y =", y)

        point = Point(x, y)

        if len(G_.nodes()) > n_nodes_for_kd:
            node_names, _ = apls_utils.nodes_near_point(
                x, y, kdtree, kd_idx_dic, x_coord=x_coord, y_coord=y_coord,
                n_neighbors=n_neighbors, verbose=False)
            nearby_nodes_set = set(node_names)
        else:
            nearby_nodes_set = set()

        Gout, node_props, xnew, ynew = insert_point_into_G(
            Gout, point, node_id=node_id,
            max_distance_meters=max_distance_meters,
            nearby_nodes_set=nearby_nodes_set,
            allow_renaming=allow_renaming,
            verbose=super_verbose)

        if x != 0 and y != 0:
            new_xs.append(xnew)
            new_ys.append(ynew)

    print("Time to run insert_control_points():", time.time() - t0, "seconds")
    return Gout, new_xs, new_ys


###############################################################################
def create_graph_midpoints(G_, linestring_delta=50, is_curved_eps=0.03,
                           n_id_add_val=1, allow_renaming=True,
                           figsize=(0, 0), verbose=False,
                           super_verbose=False):
    """
    Insert midpoint nodes along long edges.

    Arguments
    ---------
    G_ : networkx graph
    linestring_delta : float
        Spacing between injected midpoints in metres. Defaults to ``50``.
    is_curved_eps : float
        Min curvature ratio to inject points; < 0 forces injection on all
        edges. Defaults to ``0.03``.
    n_id_add_val : int
        Starting offset above max(nodes()) for new node IDs.
    allow_renaming : bool
    figsize : tuple
        If not ``(0, 0)``, draw each injected midpoint (debug).
    verbose, super_verbose : bool

    Returns
    -------
    Gout, xms, yms : tuple
    """
    if len(G_.nodes()) == 0:
        return G_, [], []

    xms, yms = [], []
    Gout = G_.copy()
    midpoint_name_val = np.max(list(G_.nodes())) + n_id_add_val
    midpoint_name_inc = 1

    for u, v, data in G_.edges(data=True):
        if 'geometry' not in data:
            continue

        linelen = data['length']
        line = data['geometry']
        xs, ys = line.xy

        # skip nearly straight lines
        minx, miny, maxx, maxy = line.bounds
        dst = scipy.spatial.distance.euclidean([minx, miny], [maxx, maxy])
        if is_curved_eps >= 0 and \
                np.abs(dst - linelen) / linelen < is_curved_eps:
            continue

        # skip very short lines
        if linelen < 0.75 * linestring_delta:
            continue

        if verbose:
            print("create_graph_midpoints() u,v:", u, v)

        if linelen <= linestring_delta:
            interp_dists = [0.5 * line.length]
        else:
            npoints = len(np.arange(0, linelen, linestring_delta)) + 1
            interp_dists = np.linspace(0, linelen, npoints)[1:-1]

        for j, d in enumerate(interp_dists):
            if verbose:
                print("  ", j, "interp_dist:", d)
            midPoint = line.interpolate(d)
            xm = float(midPoint.x)
            ym = float(midPoint.y)
            point = Point(xm, ym)
            xms.append(xm)
            yms.append(ym)

            node_id = midpoint_name_val
            midpoint_name_val += midpoint_name_inc
            if verbose:
                print("  node_id:", node_id)

            Gout, _, _, _ = insert_point_into_G(
                Gout, point, node_id=node_id, allow_renaming=allow_renaming,
                verbose=super_verbose)

        if figsize != (0, 0):
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            ax.plot(xs, ys, color='#6699cc', alpha=0.7, linewidth=3,
                    solid_capstyle='round', zorder=2)
            ax.scatter(xm, ym, color='red')
            ax.set_title('Line Midpoint')
            plt.axis('equal')

    return Gout, xms, yms


###############################################################################
def _clean_sub_graphs(G_, min_length=80, max_nodes_to_skip=100,
                      weight='length', verbose=True, super_verbose=False):
    """
    Remove connected subgraphs whose maximum path length is below min_length.

    Arguments
    ---------
    G_ : networkx graph
    min_length : float
        Minimum total path length to retain a subgraph. Defaults to ``80``.
    max_nodes_to_skip : int
        Don't check length for subgraphs larger than this. Defaults to ``100``.
    weight : str
    verbose, super_verbose : bool

    Returns
    -------
    G_ : networkx graph
    """
    if len(G_.nodes()) == 0:
        return G_

    if verbose:
        print("Running clean_sub_graphs...")

    sub_graphs = [G_.subgraph(c).copy()
                  for c in nx.connected_components(G_)]
    bad_nodes = []

    for G_sub in sub_graphs:
        if len(G_sub.nodes()) > max_nodes_to_skip:
            continue
        all_lengths = dict(
            nx.all_pairs_dijkstra_path_length(G_sub, weight=weight))
        lens = []
        for u in all_lengths:
            for uprime, vprime in all_lengths[u].items():
                lens.append(vprime)
        if np.max(lens) < min_length:
            bad_nodes.extend(G_sub.nodes())

    G_.remove_nodes_from(bad_nodes)
    if verbose:
        print(" num bad_nodes:", len(bad_nodes))
        print(" len(G'.nodes()):", len(G_.nodes()))
    return G_


###############################################################################
def _create_gt_graph(geoJson, im_test_file, network_type='all_private',
                     valid_road_types=None, osmidx=0, osmNodeidx=0,
                     subgraph_filter_weight='length', min_subgraph_length=5,
                     travel_time_key='travel_time_s',
                     speed_key='inferred_speed_mps', use_pix_coords=False,
                     verbose=False, super_verbose=False):
    """
    Load a GeoJSON and build a refined ground-truth NetworkX graph.

    Returns
    -------
    G_gt, G0gt_init : tuple
    """
    if valid_road_types is None:
        valid_road_types = set()

    t0 = time.time()
    if verbose:
        print("Executing graphTools.create_graphGeoJson()...")
    G0gt_init = graphTools.create_graphGeoJson(
        geoJson, name='unnamed', retain_all=True,
        network_type=network_type, valid_road_types=valid_road_types,
        osmidx=osmidx, osmNodeidx=osmNodeidx, verbose=verbose)
    if verbose:
        print("Time to run create_graphGeoJson():", time.time() - t0, "s")

    G_gt = _refine_gt_graph(
        G0gt_init, im_test_file,
        subgraph_filter_weight=subgraph_filter_weight,
        min_subgraph_length=min_subgraph_length,
        travel_time_key=travel_time_key, speed_key=speed_key,
        use_pix_coords=use_pix_coords,
        verbose=verbose, super_verbose=super_verbose)

    return G_gt, G0gt_init


###############################################################################
def _refine_gt_graph(G0gt_init, im_test_file,
                     subgraph_filter_weight='length', min_subgraph_length=5,
                     travel_time_key='travel_time_s',
                     speed_key='inferred_speed_mps', use_pix_coords=False,
                     verbose=False, super_verbose=False):
    """
    Refine a raw ground-truth graph: project, simplify, clean, add travel time.

    Returns
    -------
    G_gt : networkx graph
    """
    t1 = time.time()

    # attach geometry and optionally pixel geometry to each edge
    for u, v, key, data in G0gt_init.edges(keys=True, data=True):
        if 'geometry' not in data:
            sourcex, sourcey = G0gt_init.nodes[u]['x'], G0gt_init.nodes[u]['y']
            targetx, targety = G0gt_init.nodes[v]['x'], G0gt_init.nodes[v]['y']
            line_geom = LineString([Point(sourcex, sourcey),
                                    Point(targetx, targety)])
        else:
            line_geom = data['geometry']
        data['geometry_latlon'] = line_geom.wkt

        if os.path.exists(im_test_file):
            geom_pix = apls_utils.geomGeo2geomPixel(line_geom,
                                                    input_raster=im_test_file)
            data['geometry_pix'] = geom_pix.wkt
            data['length_pix'] = geom_pix.length

    if len(G0gt_init.nodes()) == 0:
        return G0gt_init

    G0gt = osmnx_funcs.project_graph(G0gt_init)
    if verbose:
        print("len G0gt.nodes():", len(G0gt.nodes()))

    if verbose:
        print("Simplifying graph...")
    try:
        G2gt_init0 = osmnx_funcs.simplify_graph(G0gt).to_undirected()
    except Exception:
        print("Cannot simplify graph, using original")
        G2gt_init0 = G0gt

    G2gt_init1 = create_edge_linestrings(G2gt_init0.copy(), remove_redundant=True)
    if verbose:
        print("Time to project/simplify/create linestrings:",
              time.time() - t1, "s")

    G2gt_init2 = _clean_sub_graphs(
        G2gt_init1.copy(), min_length=min_subgraph_length,
        weight=subgraph_filter_weight, verbose=verbose,
        super_verbose=super_verbose)

    # set pixel coords
    try:
        if os.path.exists(im_test_file):
            G_gt_almost = apls_utils._set_pix_coords(
                G2gt_init2.copy(), im_test_file)
        else:
            G_gt_almost = G2gt_init2
    except Exception:
        G_gt_almost = G2gt_init2

    # ensure correct pixel coords
    if os.path.exists(im_test_file):
        for n in G_gt_almost.nodes():
            x, y = G_gt_almost.nodes[n]['x'], G_gt_almost.nodes[n]['y']
            geom_pix = apls_utils.geomGeo2geomPixel(
                Point(x, y), input_raster=im_test_file)
            [(xp, yp)] = list(geom_pix.coords)
            G_gt_almost.nodes[n]['x_pix'] = xp
            G_gt_almost.nodes[n]['y_pix'] = yp

    # merge list-valued geometry attributes (from simplify)
    if verbose:
        print("Merge 'geometry' linestrings...")
    keys_tmp = ['geometry_pix', 'geometry_latlon']
    speed_keys = [speed_key, 'inferred_speed_mph', 'inferred_speed_mps']
    for u, v, attr_dict in G_gt_almost.edges(data=True):
        for key_tmp in keys_tmp:
            if key_tmp not in attr_dict:
                continue
            geom = attr_dict[key_tmp]
            if isinstance(geom, list):
                if geom and isinstance(geom[0], str):
                    geom = [shapely.wkt.loads(z) for z in geom]
                attr_dict[key_tmp] = shapely.ops.linemerge(geom)
            elif isinstance(geom, str):
                attr_dict[key_tmp] = shapely.wkt.loads(geom)

        if 'wkt_pix' in attr_dict:
            attr_dict['wkt_pix'] = attr_dict['geometry_pix'].wkt
        if 'length_pix' in attr_dict:
            attr_dict['length_pix'] = np.sum([attr_dict['length_pix']])

        for sk in speed_keys:
            if sk in attr_dict and isinstance(attr_dict[sk], list):
                if verbose:
                    print("  Taking mean of multiple speeds on edge:", u, v)
                attr_dict[sk] = np.mean(attr_dict[sk])

    G_gt = add_travel_time(G_gt_almost.copy(),
                           speed_key=speed_key,
                           travel_time_key=travel_time_key)
    return G_gt


###############################################################################
def make_graphs(G_gt_, G_p_,
                weight='length',
                speed_key='inferred_speed_mps',
                travel_time_key='travel_time_s',
                max_nodes_for_midpoints=500,
                linestring_delta=50,
                is_curved_eps=0.012,
                max_snap_dist=4,
                allow_renaming=True,
                verbose=False,
                super_verbose=False):
    """
    Match nodes in ground-truth and proposal graphs and compute path lengths.

    Returns
    -------
    10-tuple : (G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime,
                control_points_gt, control_points_prop,
                all_pairs_lengths_gt_native, all_pairs_lengths_prop_native,
                all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime)
    """
    t0 = time.time()
    print("Executing make_graphs()...")

    # validate weight key in gt graph
    for u, v, data in G_gt_.edges(keys=False, data=True):
        if weight not in data:
            print("Error!", weight, "not in G_gt_ edge", u, v)
            return

    # ensure geometry is a Shapely object (not a WKT string)
    for u, v, key, data in G_gt_.edges(keys=True, data=True):
        try:
            line = data['geometry']
        except KeyError:
            line = data[0]['geometry']
        if isinstance(line, str):
            data['geometry'] = shapely.wkt.loads(line)

    G_gt0 = create_edge_linestrings(G_gt_.to_undirected())

    if verbose:
        print("len G_gt.nodes():", len(list(G_gt0.nodes())))

    G_gt_cp0, xms, yms = create_graph_midpoints(
        G_gt0.copy(), linestring_delta=linestring_delta,
        is_curved_eps=is_curved_eps, verbose=False)
    G_gt_cp = add_travel_time(G_gt_cp0.copy(),
                              speed_key=speed_key,
                              travel_time_key=travel_time_key)

    control_points_gt = []
    for n in G_gt_cp.nodes():
        u_x, u_y = G_gt_cp.nodes[n]['x'], G_gt_cp.nodes[n]['y']
        control_points_gt.append([n, u_x, u_y])
    if verbose:
        print("len control_points_gt:", len(control_points_gt))

    all_pairs_lengths_gt_native = dict(
        nx.shortest_path_length(G_gt_cp, weight=weight))

    ###########################################################################
    # Proposal
    for u, v, data in G_p_.edges(keys=False, data=True):
        if weight not in data:
            print("Error!", weight, "not in G_p_ edge", u, v)
            return

    for u, v, key, data in G_p_.edges(keys=True, data=True):
        try:
            line = data['geometry']
        except KeyError:
            line = data[0]['geometry']
        if isinstance(line, str):
            data['geometry'] = shapely.wkt.loads(line)

    G_p0 = create_edge_linestrings(G_p_.to_undirected())
    G_p = add_travel_time(G_p0.copy(),
                          speed_key=speed_key,
                          travel_time_key=travel_time_key)

    G_p_cp0, xms_p, yms_p = create_graph_midpoints(
        G_p.copy(), linestring_delta=linestring_delta,
        is_curved_eps=is_curved_eps, verbose=False)
    G_p_cp = add_travel_time(G_p_cp0.copy(),
                             speed_key=speed_key,
                             travel_time_key=travel_time_key)

    control_points_prop = []
    for n in G_p_cp.nodes():
        u_x, u_y = G_p_cp.nodes[n]['x'], G_p_cp.nodes[n]['y']
        control_points_prop.append([n, u_x, u_y])

    all_pairs_lengths_prop_native = dict(
        nx.shortest_path_length(G_p_cp, weight=weight))

    ###########################################################################
    # insert GT control points into proposal
    if verbose:
        print("Inserting", len(control_points_gt), "control points into G_p...")
    G_p_cp_prime0, xn_p, yn_p = insert_control_points(
        G_p.copy(), control_points_gt,
        max_distance_meters=max_snap_dist, allow_renaming=allow_renaming,
        verbose=super_verbose)
    G_p_cp_prime = add_travel_time(G_p_cp_prime0.copy(),
                                   speed_key=speed_key,
                                   travel_time_key=travel_time_key)

    # insert proposal control points into GT
    if verbose:
        print("Inserting", len(control_points_prop),
              "control points into G_gt...")
    G_gt_cp_prime0, xn_gt, yn_gt = insert_control_points(
        G_gt_, control_points_prop,
        max_distance_meters=max_snap_dist, allow_renaming=allow_renaming,
        verbose=super_verbose)
    G_gt_cp_prime = add_travel_time(G_gt_cp_prime0.copy(),
                                    speed_key=speed_key,
                                    travel_time_key=travel_time_key)

    all_pairs_lengths_gt_prime = dict(
        nx.shortest_path_length(G_gt_cp_prime, weight=weight))
    all_pairs_lengths_prop_prime = dict(
        nx.shortest_path_length(G_p_cp_prime, weight=weight))

    print("Time to run make_graphs():", time.time() - t0, "seconds")
    return (G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime,
            control_points_gt, control_points_prop,
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native,
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime)


###############################################################################
def make_graphs_yuge(G_gt, G_p,
                     weight='length',
                     speed_key='inferred_speed_mps',
                     travel_time_key='travel_time_s',
                     max_nodes=500,
                     max_snap_dist=4,
                     allow_renaming=True,
                     verbose=True, super_verbose=False):
    """
    Match nodes in large graphs by subsampling control nodes.

    Skips midpoint injection; samples up to ``max_nodes`` nodes from each
    graph for path comparison.

    Returns
    -------
    Same 10-tuple as ``make_graphs``.
    """
    t0 = time.time()
    print("Executing make_graphs_yuge()...")

    for graph_name, G in [('G_gt', G_gt), ('G_p', G_p)]:
        for u, v, key, data in G.edges(keys=True, data=True):
            try:
                line = data['geometry']
            except KeyError:
                line = data[0]['geometry']
            if isinstance(line, str):
                data['geometry'] = shapely.wkt.loads(line)

    G_gt_cp = G_gt.to_undirected()
    if verbose:
        print("len(G_gt.nodes()):", len(G_gt_cp.nodes()))

    sample_size = min(max_nodes, len(G_gt_cp.nodes()))
    rand_nodes_gt = random.sample(list(G_gt_cp.nodes()), sample_size)
    rand_nodes_gt_set = set(rand_nodes_gt)
    control_points_gt = []
    for n in rand_nodes_gt:
        u_x, u_y = G_gt_cp.nodes[n]['x'], G_gt_cp.nodes[n]['y']
        control_points_gt.append([n, u_x, u_y])

    G_gt_cp = add_travel_time(G_gt_cp, speed_key=speed_key,
                              travel_time_key=travel_time_key)

    # all-pairs lengths for GT (restricted to rand_nodes_gt)
    tt = time.time()
    if verbose:
        print("Computing all_pairs_lengths_gt_native...")
    all_pairs_lengths_gt_native = {}
    for itmp, source in enumerate(rand_nodes_gt):
        if verbose and (itmp % 50) == 0:
            print(itmp, "source:", source)
        paths_tmp = dict(nx.single_source_dijkstra_path_length(
            G_gt_cp, source, weight=weight))
        for k in list(paths_tmp.keys()):
            if k not in rand_nodes_gt_set:
                del paths_tmp[k]
        all_pairs_lengths_gt_native[source] = paths_tmp
    if verbose:
        print("Time for gt native paths:", time.time() - tt, "s")

    G_p_cp = G_p.to_undirected()
    sample_size = min(max_nodes, len(G_p_cp.nodes()))
    rand_nodes_p = random.sample(list(G_p_cp.nodes()), sample_size)
    rand_nodes_p_set = set(rand_nodes_p)
    control_points_prop = []
    for n in rand_nodes_p:
        u_x, u_y = G_p_cp.nodes[n]['x'], G_p_cp.nodes[n]['y']
        control_points_prop.append([n, u_x, u_y])

    G_p_cp = add_travel_time(G_p_cp, speed_key=speed_key,
                             travel_time_key=travel_time_key)

    tt = time.time()
    if verbose:
        print("Computing all_pairs_lengths_prop_native...")
    all_pairs_lengths_prop_native = {}
    for itmp, source in enumerate(rand_nodes_p):
        if verbose and (itmp % 50) == 0:
            print(itmp, "source:", source)
        paths_tmp = dict(nx.single_source_dijkstra_path_length(
            G_p_cp, source, weight=weight))
        for k in list(paths_tmp.keys()):
            if k not in rand_nodes_p_set:
                del paths_tmp[k]
        all_pairs_lengths_prop_native[source] = paths_tmp
    if verbose:
        print("Time for prop native paths:", time.time() - tt, "s")

    # insert GT control points into proposal
    if verbose:
        print("Inserting", len(control_points_gt), "control points into G_p...")
    G_p_cp_prime, xn_p, yn_p = insert_control_points(
        G_p.copy(), control_points_gt, max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming, verbose=super_verbose)
    G_p_cp_prime = add_travel_time(G_p_cp_prime, speed_key=speed_key,
                                   travel_time_key=travel_time_key)

    # insert proposal control points into GT
    if verbose:
        print("Inserting", len(control_points_prop),
              "control points into G_gt...")
    G_gt_cp_prime, xn_gt, yn_gt = insert_control_points(
        G_gt, control_points_prop, max_distance_meters=max_snap_dist,
        allow_renaming=allow_renaming, verbose=super_verbose)
    G_gt_cp_prime = add_travel_time(G_gt_cp_prime, speed_key=speed_key,
                                    travel_time_key=travel_time_key)

    # paths for primed graphs (restricted to sampled node sets)
    G_gt_cp_prime_nodes_set = set(G_gt_cp_prime.nodes())
    tt = time.time()
    all_pairs_lengths_gt_prime = {}
    if verbose:
        print("Computing all_pairs_lengths_gt_prime...")
    for itmp, source in enumerate(rand_nodes_p_set):
        if verbose and (itmp % 50) == 0:
            print(itmp, "source:", source)
        if source in G_gt_cp_prime_nodes_set:
            paths_tmp = dict(nx.single_source_dijkstra_path_length(
                G_gt_cp_prime, source, weight=weight))
            for k in list(paths_tmp.keys()):
                if k not in rand_nodes_p_set:
                    del paths_tmp[k]
            all_pairs_lengths_gt_prime[source] = paths_tmp
    if verbose:
        print("Time for gt prime paths:", time.time() - tt, "s")

    G_p_cp_prime_nodes_set = set(G_p_cp_prime.nodes())
    tt = time.time()
    all_pairs_lengths_prop_prime = {}
    if verbose:
        print("Computing all_pairs_lengths_prop_prime...")
    for itmp, source in enumerate(rand_nodes_gt_set):
        if verbose and (itmp % 50) == 0:
            print(itmp, "source:", source)
        if source in G_p_cp_prime_nodes_set:
            paths_tmp = dict(nx.single_source_dijkstra_path_length(
                G_p_cp_prime, source, weight=weight))
            for k in list(paths_tmp.keys()):
                if k not in rand_nodes_gt_set:
                    del paths_tmp[k]
            all_pairs_lengths_prop_prime[source] = paths_tmp
    if verbose:
        print("Time for prop prime paths:", time.time() - tt, "s")

    print("Time to run make_graphs_yuge():", time.time() - t0, "seconds")
    return (G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime,
            control_points_gt, control_points_prop,
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native,
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime)


###############################################################################
def single_path_metric(len_gt, len_prop, diff_max=1):
    """
    Compute the APLS metric for a single path pair.

    Returns ``1 - min(|len_gt - len_prop| / len_gt, diff_max)``.
    If ``len_prop < 0`` (missing path) returns ``diff_max``.

    Arguments
    ---------
    len_gt : float
    len_prop : float
    diff_max : float

    Returns
    -------
    metric : float in [0, diff_max]
    """
    if len_gt <= 0:
        return 0
    if len_prop < 0:
        return diff_max
    return np.min([diff_max, np.abs(len_gt - len_prop) / len_gt])


###############################################################################
def path_sim_metric(all_pairs_lengths_gt, all_pairs_lengths_prop,
                    control_nodes=None, min_path_length=10,
                    diff_max=1, missing_path_len=-1, normalize=True,
                    verbose=False):
    """
    Compute APLS over a set of paths.

    Arguments
    ---------
    all_pairs_lengths_gt : dict
    all_pairs_lengths_prop : dict
    control_nodes : list or None
        Nodes to evaluate; if None/empty, uses all keys in gt dict.
    min_path_length : float
        Skip paths shorter than this. Defaults to ``10``.
    diff_max : float
    missing_path_len : float
        Sentinel for a missing path. Defaults to ``-1``.
    normalize : bool
    verbose : bool

    Returns
    -------
    C, diffs, routes, diff_dic : tuple
    """
    if control_nodes is None:
        control_nodes = []

    diffs = []
    routes = []
    diff_dic = {}
    gt_start_nodes_set = set(all_pairs_lengths_gt.keys())
    prop_start_nodes_set = set(all_pairs_lengths_prop.keys())
    t0 = time.time()

    if not gt_start_nodes_set:
        return 0, [], [], {}

    good_nodes = list(all_pairs_lengths_gt.keys()) \
        if not control_nodes else control_nodes

    if verbose:
        print("Computing path_sim_metric()... good_nodes:", good_nodes)

    for start_node in good_nodes:
        node_dic_tmp = {}

        if start_node not in gt_start_nodes_set:
            print("node", start_node, "not in gt set, skipping")
            for end_node, len_prop in all_pairs_lengths_prop.get(
                    start_node, {}).items():
                diffs.append(diff_max)
                routes.append([start_node, end_node])
                node_dic_tmp[end_node] = diff_max
            continue

        paths = all_pairs_lengths_gt[start_node]

        # start node missing from proposal → all paths from it score diff_max
        if start_node not in prop_start_nodes_set:
            for end_node, len_gt in paths.items():
                if end_node != start_node and end_node in good_nodes:
                    diffs.append(diff_max)
                    routes.append([start_node, end_node])
                    node_dic_tmp[end_node] = diff_max
            diff_dic[start_node] = node_dic_tmp
            continue

        paths_prop = all_pairs_lengths_prop[start_node]
        end_nodes_gt_set = set(paths.keys()) & set(good_nodes)
        end_nodes_prop_set = set(paths_prop.keys())

        for end_node in end_nodes_gt_set:
            len_gt = paths[end_node]
            if len_gt < min_path_length:
                continue
            len_prop = paths_prop.get(end_node, missing_path_len)
            if verbose:
                print("end_node:", end_node, "len_gt:", len_gt,
                      "len_prop:", len_prop)
            diff = single_path_metric(len_gt, len_prop, diff_max=diff_max)
            diffs.append(diff)
            routes.append([start_node, end_node])
            node_dic_tmp[end_node] = diff

        diff_dic[start_node] = node_dic_tmp

    if not diffs:
        return 0, [], [], {}

    diff_tot = np.sum(diffs)
    if normalize:
        C = 1.0 - diff_tot / len(diffs)
    else:
        C = diff_tot

    print("Time to compute metric (score =", C, ") for",
          len(diffs), "routes:", time.time() - t0, "seconds")
    return C, diffs, routes, diff_dic


###############################################################################
def compute_apls_metric(all_pairs_lengths_gt_native,
                        all_pairs_lengths_prop_native,
                        all_pairs_lengths_gt_prime,
                        all_pairs_lengths_prop_prime,
                        control_points_gt, control_points_prop,
                        res_dir='', min_path_length=10,
                        verbose=False, super_verbose=False):
    """
    Compute bidirectional APLS metric and optionally save plots.

    Returns
    -------
    C_tot, C_gt_onto_prop, C_prop_onto_gt : tuple
        C_tot is the harmonic mean of the two directional scores.
    """
    t0 = time.time()

    if not all_pairs_lengths_gt_native or not all_pairs_lengths_prop_native:
        print("Empty path lengths, returning 0")
        return 0, 0, 0

    # GT snapped onto proposal
    print("Compute metric (gt snapped onto prop)")
    control_nodes = [z[0] for z in control_points_gt]
    if verbose:
        print("control_nodes_gt:", control_nodes)
    C_gt_onto_prop, diffs, routes, diff_dic = path_sim_metric(
        all_pairs_lengths_gt_native,
        all_pairs_lengths_prop_prime,
        control_nodes=control_nodes,
        min_path_length=min_path_length,
        diff_max=1, missing_path_len=-1, normalize=True,
        verbose=super_verbose)
    dt1 = time.time() - t0

    if res_dir:
        scatter_png = os.path.join(
            res_dir, 'all_pairs_paths_diffs_gt_to_prop.png')
        hist_png = os.path.join(
            res_dir, 'all_pairs_paths_diffs_hist_gt_to_prop.png')
        routes_str = [] if len(routes) > 100 else \
            [str(z[0]) + '-' + str(z[1]) for z in routes]
        apls_plots.plot_metric(
            C_gt_onto_prop, diffs, routes_str=routes_str,
            figsize=(10, 5), scatter_alpha=0.8, scatter_size=8,
            scatter_png=scatter_png, hist_png=hist_png)

    # Proposal snapped onto GT
    print("Compute metric (prop snapped onto gt)")
    t1 = time.time()
    control_nodes = [z[0] for z in control_points_prop]
    if verbose:
        print("control_nodes:", control_nodes)
    C_prop_onto_gt, diffs, routes, diff_dic = path_sim_metric(
        all_pairs_lengths_prop_native,
        all_pairs_lengths_gt_prime,
        control_nodes=control_nodes,
        min_path_length=min_path_length,
        diff_max=1, missing_path_len=-1, normalize=True,
        verbose=super_verbose)
    dt2 = time.time() - t1

    if res_dir:
        scatter_png = os.path.join(
            res_dir, 'all_pairs_paths_diffs_prop_to_gt.png')
        hist_png = os.path.join(
            res_dir, 'all_pairs_paths_diffs_hist_prop_to_gt.png')
        routes_str = [] if len(routes) > 100 else \
            [str(z[0]) + '-' + str(z[1]) for z in routes]
        apls_plots.plot_metric(
            C_prop_onto_gt, diffs, routes_str=routes_str,
            figsize=(10, 5), scatter_alpha=0.8, scatter_size=8,
            scatter_png=scatter_png, hist_png=hist_png)

    print("C_gt_onto_prop, C_prop_onto_gt:", C_gt_onto_prop, C_prop_onto_gt)
    if C_gt_onto_prop <= 0 or C_prop_onto_gt <= 0 \
            or np.isnan(C_gt_onto_prop) or np.isnan(C_prop_onto_gt):
        C_tot = 0
    else:
        C_tot = scipy.stats.hmean([C_gt_onto_prop, C_prop_onto_gt])
        if np.isnan(C_tot):
            C_tot = 0

    print("Total APLS Metric = Mean(",
          np.round(C_gt_onto_prop, 2), "+",
          np.round(C_prop_onto_gt, 2), ") =", np.round(C_tot, 2))
    print("Total time to compute metric:", dt1 + dt2, "seconds")
    return C_tot, C_gt_onto_prop, C_prop_onto_gt


###############################################################################
def gather_files(test_method, truth_dir, prop_dir,
                 im_dir='', im_prefix='',
                 gt_wkt_file='', prop_wkt_file='',
                 max_files=1000,
                 gt_subgraph_filter_weight='length',
                 gt_min_subgraph_length=5,
                 prop_subgraph_filter_weight='length_pix',
                 prop_min_subgraph_length=10,
                 use_pix_coords=True,
                 speed_key='inferred_speed_mps',
                 travel_time_key='travel_time_s',
                 wkt_weight_key='travel_time_s',
                 default_speed=13.41,
                 verbose=False, super_verbose=False):
    """
    Build matched lists of ground-truth and proposal graphs.

    test_method options:
        ``gt_pkl_prop_pkl``   GT=pickle, Proposal=pickle
        ``gt_json_prop_pkl``  GT=GeoJSON, Proposal=pickle
        ``gt_json_prop_json`` GT=GeoJSON, Proposal=GeoJSON
        ``gt_json_prop_wkt``  GT=GeoJSON, Proposal=WKT CSV
        ``gt_wkt_prop_wkt``   GT=WKT CSV, Proposal=WKT CSV

    Returns
    -------
    gt_list, gp_list, root_list, im_loc_list : tuple
    """
    print("Gathering files...")
    gt_list, gp_list, root_list, im_loc_list = [], [], [], []

    # ------------------------------------------------------------------
    if test_method == 'gt_pkl_prop_pkl':
        name_list = sorted(os.listdir(truth_dir))
        for i, f in enumerate(name_list):
            if i >= max_files:
                break
            if not f.endswith(('.gpickle', '.pkl')):
                continue

            outroot = f.split('.')[0]
            gt_file = os.path.join(truth_dir, f)
            prop_file = os.path.join(prop_dir, outroot + '.gpickle')
            if not os.path.exists(prop_file):
                prop_file = os.path.join(
                    prop_dir, 'fold0_RGB-PanSharpen_' + outroot + '.gpickle')
            im_file = os.path.join(im_dir, outroot + '.tif')
            if not os.path.exists(prop_file):
                print("missing prop file:", prop_file)
                continue

            G_gt_init = _ensure_edge_speed(
                _read_graph(gt_file), speed_key=speed_key,
                default_speed=default_speed)
            G_p_init = _ensure_edge_speed(
                _read_graph(prop_file), speed_key=speed_key,
                default_speed=default_speed)

            gt_list.append(G_gt_init)
            gp_list.append(G_p_init)
            root_list.append(outroot)
            im_loc_list.append(im_file)

    # ------------------------------------------------------------------
    elif test_method == 'gt_json_prop_pkl':
        valid_road_types = set()
        name_list = sorted(os.listdir(truth_dir))
        for i, f in enumerate(name_list):
            if not f.endswith('.geojson'):
                continue
            if i >= max_files:
                break

            if f.startswith('spacenetroads'):
                outroot = f.split('spacenetroads_')[-1].split('.')[0]
                prop_file = os.path.join(
                    prop_dir, im_prefix + outroot + '.gpickle')
                im_file = os.path.join(
                    im_dir, im_prefix + outroot + '.tif')
            elif f.startswith('osmroads'):
                outroot = f.split('osmroads_')[-1].split('.')[0]
                prop_file = os.path.join(
                    prop_dir, im_prefix + outroot + '.gpickle')
                im_file = os.path.join(
                    im_dir, im_prefix + outroot + '.tif')
            elif f.startswith('SN'):
                outroot = f.split('.')[0].replace('geojson_roads_speed_', '')
                prop_file = os.path.join(
                    prop_dir,
                    f.replace('geojson_roads_speed', 'PS-RGB')
                ).replace('.geojson', '.gpickle')
                im_file = os.path.join(
                    im_dir,
                    f.split('.')[0].replace('geojson_roads_speed', 'PS-RGB')
                    + '.tif')
            else:
                print("Unknown GeoJSON naming convention:", f)
                continue

            print("\n", i, "outroot:", outroot)
            gt_file = os.path.join(truth_dir, f)
            if not os.path.exists(prop_file):
                print("prop file DNE, skipping:", prop_file)
                continue

            osmidx, osmNodeidx = 10000, 10000
            G_gt_init, _ = _create_gt_graph(
                gt_file, im_file, network_type='all_private',
                valid_road_types=valid_road_types,
                subgraph_filter_weight=gt_subgraph_filter_weight,
                min_subgraph_length=gt_min_subgraph_length,
                use_pix_coords=use_pix_coords,
                osmidx=osmidx, osmNodeidx=osmNodeidx,
                speed_key=speed_key,
                travel_time_key=travel_time_key, verbose=verbose)
            if len(G_gt_init.nodes()) == 0:
                continue

            G_p_init = _ensure_edge_speed(
                _read_graph(prop_file), speed_key=speed_key,
                default_speed=default_speed)

            gt_list.append(G_gt_init)
            gp_list.append(G_p_init)
            root_list.append(outroot)
            im_loc_list.append(im_file)

    # ------------------------------------------------------------------
    if test_method == 'gt_json_prop_json':
        valid_road_types = set()
        for f in os.listdir(truth_dir):
            if not f.endswith('.geojson'):
                continue
            outroot = f.split('.')[0]
            gt_file = os.path.join(truth_dir, f)
            prop_file = os.path.join(prop_dir, outroot + '.geojson')
            if not os.path.exists(prop_file):
                prop_file = os.path.join(prop_dir, outroot + 'prop.geojson')
            im_file = ''

            osmidx, osmNodeidx = 0, 0
            G_gt_init, _ = _create_gt_graph(
                gt_file, im_file, network_type='all_private',
                valid_road_types=valid_road_types,
                use_pix_coords=use_pix_coords,
                osmidx=osmidx, osmNodeidx=osmNodeidx,
                speed_key=speed_key,
                travel_time_key=travel_time_key, verbose=verbose)
            if len(G_gt_init.nodes()) == 0:
                continue

            osmidx, osmNodeidx = 500, 500
            G_p_init, _ = _create_gt_graph(
                prop_file, im_file, network_type='all_private',
                valid_road_types=valid_road_types,
                use_pix_coords=use_pix_coords,
                osmidx=osmidx, osmNodeidx=osmNodeidx, verbose=verbose)

            gt_list.append(G_gt_init)
            gp_list.append(G_p_init)
            root_list.append(outroot)

    # ------------------------------------------------------------------
    elif test_method == 'gt_json_prop_wkt':
        valid_road_types = set()
        name_list = sorted(os.listdir(truth_dir))
        for i, f in enumerate(name_list):
            if not f.endswith('.geojson'):
                continue
            if i >= max_files:
                break

            outroot = f.split('spacenetroads_')[-1].split('.')[0]
            if verbose:
                print("\n", i, "outroot:", outroot)
            gt_file = os.path.join(truth_dir, f)
            im_file = os.path.join(
                im_dir,
                f.split('.')[0].replace('geojson_roads_speed', 'PS-RGB')
                + '.tif')

            osmidx, osmNodeidx = 0, 0
            G_gt_init, _ = _create_gt_graph(
                gt_file, im_file, network_type='all_private',
                valid_road_types=valid_road_types,
                subgraph_filter_weight=gt_subgraph_filter_weight,
                min_subgraph_length=gt_min_subgraph_length,
                use_pix_coords=use_pix_coords,
                osmidx=osmidx, osmNodeidx=osmNodeidx,
                speed_key=speed_key,
                travel_time_key=travel_time_key, verbose=verbose)
            if len(G_gt_init.nodes()) == 0:
                continue

            df_wkt = pd.read_csv(prop_wkt_file)
            AOI_root = 'AOI' + f.split('AOI')[-1]
            image_id = AOI_root.split('.')[0].replace(
                'geojson_roads_speed_', '')
            print("image_id", image_id)

            df_filt = df_wkt['WKT_Pix'][df_wkt['ImageId'] == image_id]
            wkt_list = df_filt.values
            weight_list = df_wkt[wkt_weight_key][
                df_wkt['ImageId'] == image_id].values

            if len(wkt_list) == 0 or wkt_list[0] == 'LINESTRING EMPTY':
                continue

            node_iter, edge_iter = 1000, 1000
            G_p_init = wkt_to_G.wkt_to_G(
                wkt_list, weight_list=weight_list, im_file=im_file,
                prop_subgraph_filter_weight=prop_subgraph_filter_weight,
                min_subgraph_length=prop_min_subgraph_length,
                node_iter=node_iter, edge_iter=edge_iter,
                verbose=super_verbose)

            if 'time' in wkt_weight_key:
                for u, v, data in G_p_init.edges(data=True):
                    data[travel_time_key] = data['weight']
            else:
                G_p_init = add_travel_time(
                    G_p_init, speed_key=speed_key,
                    travel_time_key=travel_time_key,
                    default_speed=default_speed)

            gt_list.append(G_gt_init)
            gp_list.append(G_p_init)
            root_list.append(outroot)
            im_loc_list.append(im_file)

    # ------------------------------------------------------------------
    elif test_method == 'gt_wkt_prop_wkt':
        im_list = sorted(os.listdir(im_dir))
        for i, f in enumerate(im_list):
            if not f.endswith('.tif'):
                continue
            if i >= max_files:
                break

            im_file = os.path.join(im_dir, f)
            outroot = ('AOI'
                       + f.split('.')[0].split('AOI')[-1]
                       .replace('PS-RGB_', ''))

            df_wkt_gt = pd.read_csv(gt_wkt_file)
            image_id = outroot

            df_filt = df_wkt_gt['WKT_Pix'][df_wkt_gt['ImageId'] == image_id]
            wkt_list = df_filt.values
            weight_list = df_wkt_gt[wkt_weight_key][
                df_wkt_gt['ImageId'] == image_id].values

            if len(wkt_list) == 0 or wkt_list[0] == 'LINESTRING EMPTY':
                continue

            node_iter, edge_iter = 10000, 10000
            G_gt_init0 = wkt_to_G.wkt_to_G(
                wkt_list, weight_list=weight_list, im_file=im_file,
                prop_subgraph_filter_weight=prop_subgraph_filter_weight,
                min_subgraph_length=prop_min_subgraph_length,
                node_iter=node_iter, edge_iter=edge_iter,
                verbose=super_verbose)
            G_gt_init = _refine_gt_graph(
                G_gt_init0, im_file,
                subgraph_filter_weight=gt_subgraph_filter_weight,
                min_subgraph_length=gt_min_subgraph_length,
                travel_time_key=travel_time_key, speed_key=speed_key,
                use_pix_coords=use_pix_coords, verbose=verbose,
                super_verbose=super_verbose)

            if 'time' in wkt_weight_key:
                for u, v, data in G_gt_init.edges(data=True):
                    data[travel_time_key] = data['weight']
            else:
                G_gt_init = add_travel_time(
                    G_gt_init, speed_key=speed_key,
                    travel_time_key=travel_time_key,
                    default_speed=default_speed)

            if len(G_gt_init.nodes()) == 0:
                continue

            df_wkt = pd.read_csv(prop_wkt_file)
            df_filt = df_wkt['WKT_Pix'][df_wkt['ImageId'] == image_id]
            wkt_list = df_filt.values
            weight_list = df_wkt[wkt_weight_key][
                df_wkt['ImageId'] == image_id].values

            if len(wkt_list) == 0 or wkt_list[0] == 'LINESTRING EMPTY':
                continue

            node_iter, edge_iter = 1000, 1000
            G_p_init = wkt_to_G.wkt_to_G(
                wkt_list, weight_list=weight_list, im_file=im_file,
                prop_subgraph_filter_weight=prop_subgraph_filter_weight,
                min_subgraph_length=prop_min_subgraph_length,
                node_iter=node_iter, edge_iter=edge_iter,
                verbose=super_verbose)

            if 'time' in wkt_weight_key:
                for u, v, data in G_p_init.edges(data=True):
                    data[travel_time_key] = data['weight']
            else:
                G_p_init = add_travel_time(
                    G_p_init, speed_key=speed_key,
                    travel_time_key=travel_time_key,
                    default_speed=default_speed)

            gt_list.append(G_gt_init)
            gp_list.append(G_p_init)
            root_list.append(outroot)
            im_loc_list.append(im_file)

    return gt_list, gp_list, root_list, im_loc_list


###############################################################################
def execute(output_name, gt_list, gp_list, root_list, im_loc_list=None,
            weight='length',
            speed_key='inferred_speed_mps',
            travel_time_key='travel_time_s',
            test_method='gt_json_prop_json',
            output_dir='.',
            max_files=1000,
            linestring_delta=50,
            is_curved_eps=1e3,
            max_snap_dist=4,
            max_nodes=500,
            n_plots=10,
            min_path_length=10,
            topo_hole_size=4,
            topo_subgraph_radius=150,
            topo_interval=30,
            sp_length_buffer=0.05,
            use_pix_coords=False,
            allow_renaming=True,
            verbose=True,
            super_verbose=False):
    """
    Compute APLS, TOPO, and SP metrics for all graphs in gt_list / gp_list.

    Results are written to CSV and per-tile text files under output_dir.

    Arguments
    ---------
    output_name : str
        Subdirectory name for outputs.
    output_dir : str
        Base output directory. Defaults to ``'.'``.
    (remaining arguments documented in ``gather_files`` / ``make_graphs``)

    Returns
    -------
    None
    """
    if im_loc_list is None:
        im_loc_list = []

    C_arr = [["outroot", "APLS", "APLS_gt_onto_prop", "APLS_prop_onto_gt",
              "topo_tp_tot", "topo_fp_tot", "topo_fn_tot", "topo_precision",
              "topo_recall", "topo_f1", "sp_metric",
              "tot_meters_gt", "tot_meters_p"]]

    # plot settings
    title_fontsize = 4
    dpi = 200
    show_plots = False
    show_node_ids = True
    fig_height, fig_width = 6, 6
    route_linewidth = 4
    source_color = 'red'
    target_color = 'green'

    outdir_base2 = os.path.join(
        output_dir, str(output_name), 'weight=' + str(weight), test_method)
    os.makedirs(outdir_base2, exist_ok=True)

    t0 = time.time()
    for i, (outroot, G_gt_init, G_p_init) in enumerate(
            zip(root_list, gt_list, gp_list)):
        if i >= max_files:
            break

        im_loc = im_loc_list[i] if im_loc_list else ''
        print("\n\n\n", i+1, "/", len(root_list), "Computing:", outroot)
        t1 = time.time()

        print("len(G_gt_init.nodes():", len(G_gt_init.nodes()))
        print("len(G_p_init.nodes():", len(G_p_init.nodes()))

        # per-tile output directory
        outdir = os.path.join(outdir_base2, outroot)
        os.makedirs(outdir, exist_ok=True)

        print("\nMake gt, prop graphs...")
        if len(G_gt_init.nodes()) < 500:
            result = make_graphs(
                G_gt_init, G_p_init,
                weight=weight, speed_key=speed_key,
                travel_time_key=travel_time_key,
                linestring_delta=linestring_delta,
                is_curved_eps=is_curved_eps,
                max_snap_dist=max_snap_dist,
                allow_renaming=allow_renaming, verbose=verbose)
        else:
            result = make_graphs_yuge(
                G_gt_init, G_p_init,
                weight=weight, speed_key=speed_key,
                travel_time_key=travel_time_key,
                max_nodes=max_nodes,
                max_snap_dist=max_snap_dist,
                allow_renaming=allow_renaming, verbose=verbose,
                super_verbose=super_verbose)

        if result is None:
            continue

        (G_gt_cp, G_p_cp, G_gt_cp_prime, G_p_cp_prime,
         control_points_gt, control_points_prop,
         all_pairs_lengths_gt_native, all_pairs_lengths_prop_native,
         all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime) = result

        # APLS
        res_dir = outdir if i < n_plots else ''
        C, C_gt_onto_prop, C_prop_onto_gt = compute_apls_metric(
            all_pairs_lengths_gt_native, all_pairs_lengths_prop_native,
            all_pairs_lengths_gt_prime, all_pairs_lengths_prop_prime,
            control_points_gt, control_points_prop,
            min_path_length=min_path_length,
            verbose=verbose, res_dir=res_dir)
        print("APLS Metric =", C)

        # TOPO
        print("\nComputing TOPO Metric...")
        topo_vals = topo_metric.compute_topo(
            G_gt_init, G_p_init,
            subgraph_radius=topo_subgraph_radius,
            interval=topo_interval, hole_size=topo_hole_size,
            n_measurement_nodes=max_nodes,
            x_coord='x', y_coord='y',
            allow_multi_hole=False, make_plots=False, verbose=False)
        (topo_tp_tot, topo_fp_tot, topo_fn_tot,
         topo_precision, topo_recall, topo_f1) = topo_vals
        print("TOPO Metric =", topo_vals)

        # SP
        print("\nComputing SP Metric...")
        _, sp = sp_metric.compute_sp(
            G_gt_init, G_p_init,
            x_coord='x', y_coord='y', weight=weight,
            query_radius=max_snap_dist, length_buffer=sp_length_buffer,
            n_routes=max_nodes, verbose=False, make_plots=False)
        print("SP Metric =", sp)

        tot_meters_gt = sum(d['length']
                            for u, v, d in G_gt_init.edges(data=True))
        tot_meters_p = sum(d['length']
                           for u, v, d in G_p_init.edges(data=True))
        print("GT total edge km:", tot_meters_gt / 1000)
        print("Prop total edge km:", tot_meters_p / 1000)

        # per-tile score file
        score_file = os.path.join(
            outdir_base2,
            str(output_name) + '_weight=' + str(weight) + '_' + test_method
            + '_output__max_snap='
            + str(np.round(max_snap_dist, 2)) + 'm'
            + '_hole=' + str(np.round(topo_hole_size, 2)) + 'm.txt')
        with open(score_file, 'w') as f:
            f.write("GT Nodes Snapped Onto Proposal: " +
                    str(C_gt_onto_prop) + "\n")
            f.write("Proposal Nodes Snapped Onto GT: " +
                    str(C_prop_onto_gt) + "\n")
            f.write("Total APLS Score: " + str(C) + "\n")
            f.write("TOPO vals: " + str(topo_vals) + "\n")
            f.write("SP: " + str(sp) + "\n")

        t2 = time.time()
        print("Total time for tile:", t2 - t1, "seconds")

        C_arr.append([outroot, C, C_gt_onto_prop, C_prop_onto_gt,
                      topo_tp_tot, topo_fp_tot, topo_fn_tot,
                      topo_precision, topo_recall, topo_f1,
                      sp, tot_meters_gt, tot_meters_p])

        # plots
        if i < n_plots:
            if len(G_gt_cp.nodes()) == 0 or len(G_p_cp.nodes()) == 0:
                continue

            max_extent = max(fig_height, fig_width)
            xmin, xmax, ymin, ymax, dx, dy = apls_utils._get_graph_extent(
                G_gt_cp)
            if dx <= dy:
                fig_height = max_extent
                fig_width = max(1, max_extent * dx / dy)
            else:
                fig_width = max_extent
                fig_height = max(1, max_extent * dy / dx)

            for G_tmp, fname, title in [
                (G_gt_init, 'gt_graph.png', 'Ground Truth Graph'),
                (G_gt_cp, 'gt_graph_midpoints.png',
                 'Ground Truth With Midpoints'),
                (G_gt_cp_prime, 'gt_graph_prop_control_points.png',
                 'GT With Proposal Control Nodes'),
                (G_p_init, 'prop_graph.png', 'Proposal Graph'),
                (G_p_cp, 'prop_graph_midpoints.png', 'Proposal With Midpoints'),
                (G_p_cp_prime, 'prop_graph_midpoints_gt_control_points.png',
                 'Proposal With GT Control Nodes'),
            ]:
                fig, ax = osmnx_funcs.plot_graph(
                    G_tmp, show=show_plots, close=False,
                    fig_height=fig_height, fig_width=fig_width)
                if show_node_ids:
                    ax = apls_plots.plot_node_ids(G_tmp, ax, fontsize=4)
                ax.set_title(title, fontsize=title_fontsize)
                plt.savefig(os.path.join(outdir, fname), dpi=dpi)
                plt.close('all')

            # overlay on image
            if im_loc and os.path.exists(im_loc):
                figname = os.path.join(outdir, 'overlaid.png')
                _ = apls_plots._plot_gt_prop_graphs(
                    G_gt_init, G_p_init, im_loc,
                    figsize=(16, 8), show_endnodes=True,
                    width_key=2, width_mult=1,
                    gt_color='cyan', prop_color='lime',
                    default_node_size=20,
                    title=outroot, adjust=False,
                    figname=figname, verbose=super_verbose)

    # aggregate CSV
    if len(C_arr) > 1:
        try:
            means = np.mean(
                np.array(C_arr)[1:, 1:].astype(float), axis=0)
            C_arr.append(['means'] + list(means))
            stds = np.std(
                np.array(C_arr)[1:, 1:].astype(float), axis=0)
            C_arr.append(['stds'] + list(stds))
        except Exception:
            pass

    path_csv = os.path.join(
        outdir_base2,
        'scores__max_snap=' + str(np.round(max_snap_dist, 2)) + 'm'
        + '_hole=' + str(np.round(topo_hole_size, 2)) + 'm.csv')
    df = pd.DataFrame(C_arr[1:], columns=C_arr[0])
    df.to_csv(path_csv)
    print("Saved scores to:", path_csv)
    print("N input images:", len(root_list))
    tf = time.time()
    print("Time to compute metric:", tf - t0, "seconds")


###############################################################################
def main():
    """CLI entry point for the APLS metric."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_name', default='apls_test0',
                        help='Output folder name')
    parser.add_argument('--test_method', default='gt_pkl_prop_pkl',
                        help='Input mode (gt_pkl_prop_pkl, gt_json_prop_pkl, '
                             'gt_json_prop_json, gt_json_prop_wkt, '
                             'gt_wkt_prop_wkt)')
    parser.add_argument('--truth_dir', default='',
                        help='Ground truth files directory')
    parser.add_argument('--prop_dir', default='',
                        help='Proposal files directory')
    parser.add_argument('--im_dir', default='',
                        help='Image files directory')
    parser.add_argument('--im_prefix', default='RGB-PanSharpen_',
                        help='Image filename prefix')
    parser.add_argument('--gt_wkt_file', default='',
                        help='Ground truth WKT CSV path')
    parser.add_argument('--prop_wkt_file', default='',
                        help='Proposal WKT CSV path')
    parser.add_argument('--output_dir', default='.',
                        help='Base output directory')
    parser.add_argument('--max_snap_dist', default=4, type=int)
    parser.add_argument('--topo_hole_size', default=4, type=float)
    parser.add_argument('--topo_subgraph_radius', default=150, type=float)
    parser.add_argument('--topo_interval', default=30, type=float)
    parser.add_argument('--sp_length_buffer', default=0.05, type=float)
    parser.add_argument('--linestring_delta', default=50, type=int)
    parser.add_argument('--is_curved_eps', default=-1, type=float)
    parser.add_argument('--min_path_length', default=0.001, type=float)
    parser.add_argument('--max_nodes', default=1000, type=int)
    parser.add_argument('--max_files', default=100, type=int)
    parser.add_argument('--weight', default='length')
    parser.add_argument('--speed_key', default='inferred_speed_mps')
    parser.add_argument('--travel_time_key', default='travel_time_s')
    parser.add_argument('--wkt_weight_key', default='travel_time_s')
    parser.add_argument('--default_speed', default=13.41, type=float)
    parser.add_argument('--n_plots', default=12, type=int)
    parser.add_argument('--use_pix_coords', default=1, type=int)
    parser.add_argument('--allow_renaming', default=1, type=int)

    args = parser.parse_args()
    args.gt_subgraph_filter_weight = 'length'
    args.gt_min_subgraph_length = 5
    args.prop_subgraph_filter_weight = 'length_pix'
    args.prop_min_subgraph_length = 10

    verbose = True
    super_verbose = False

    gt_list, gp_list, root_list, im_loc_list = gather_files(
        args.test_method, args.truth_dir, args.prop_dir,
        im_dir=args.im_dir, im_prefix=args.im_prefix,
        gt_wkt_file=args.gt_wkt_file, prop_wkt_file=args.prop_wkt_file,
        max_files=args.max_files,
        gt_subgraph_filter_weight=args.gt_subgraph_filter_weight,
        gt_min_subgraph_length=args.gt_min_subgraph_length,
        prop_subgraph_filter_weight=args.prop_subgraph_filter_weight,
        prop_min_subgraph_length=args.prop_min_subgraph_length,
        use_pix_coords=bool(args.use_pix_coords),
        speed_key=args.speed_key, travel_time_key=args.travel_time_key,
        wkt_weight_key=args.wkt_weight_key,
        default_speed=args.default_speed,
        verbose=verbose, super_verbose=super_verbose)

    execute(
        args.output_name, gt_list, gp_list, root_list,
        im_loc_list=im_loc_list,
        test_method=args.test_method,
        weight=args.weight,
        speed_key=args.speed_key,
        travel_time_key=args.travel_time_key,
        output_dir=args.output_dir,
        max_files=args.max_files,
        linestring_delta=args.linestring_delta,
        is_curved_eps=args.is_curved_eps,
        max_snap_dist=args.max_snap_dist,
        max_nodes=args.max_nodes,
        n_plots=args.n_plots,
        min_path_length=args.min_path_length,
        topo_hole_size=args.topo_hole_size,
        topo_subgraph_radius=args.topo_subgraph_radius,
        topo_interval=args.topo_interval,
        sp_length_buffer=args.sp_length_buffer,
        use_pix_coords=bool(args.use_pix_coords),
        allow_renaming=bool(args.allow_renaming),
        verbose=verbose, super_verbose=super_verbose)


###############################################################################
if __name__ == "__main__":
    main()
