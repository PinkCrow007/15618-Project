# Version 10: Visualize with context
import pandas as pd
import numpy as np
import csv
import pymetis
from tqdm import tqdm
import igraph
import matplotlib.pyplot as plt
import networkx as nx
import math
import contextily as ctx
from sklearn import cluster
import os

cluster_nums = [2, 4, 8]
# Input/Output paths
base_dir = '/data_2/tyh/LPSim'
data_dir = base_dir + '/LivingCity/berkeley_2018/full_network/'
input_route = base_dir + '/LivingCity/0_route7to8.csv'
output_dir = base_dir + '/partition_results'

def calculate_cluster_center(positions, reverse_node_dict, membership, num_class):
    categorized_positions = [[] for _ in range(num_class)]
    cluster_centers = []
    for index, category in enumerate(membership):
        categorized_positions[category].append(positions[reverse_node_dict[index]])
    for category_positions in categorized_positions:
        xs = [category_position[0] for category_position in category_positions]
        ys = [category_position[1] for category_position in category_positions]
        x_mean = sum(xs)/len(xs)
        y_mean = sum(ys)/len(ys)
        cluster_centers.append([x_mean, y_mean])
    return cluster_centers

def get_full_membership(positions, nodes_dict, membership, cluster_centers):
    full_membership = []
    for index, position in positions.items():
        if index in nodes_dict.keys():
            full_membership.append(membership[nodes_dict[index]])
        else:
            current_x = position[0]
            current_y = position[1]
            current_closest = 0
            closest_distance = 10000000000000
            for center_index, center in enumerate(cluster_centers):
                center_x = center[0]
                center_y = center[1]
                current_distance = math.sqrt(math.pow(center_x-current_x, 2)+math.pow(center_y-current_y, 2))
                if current_distance < closest_distance:
                    current_closest = center_index
                    closest_distance = current_distance
            full_membership.append(current_closest)
    return full_membership

def get_color(category):
    colors = ['bisque', 'palegreen', 'skyblue', 'lightcoral', 'pink', 'thistle', 'lightsalmon', 'khaki']
    return colors[category]

def run_two_algorithms(cluster_num):
    # * Read info
    nodes_df = pd.read_csv(data_dir + 'nodes.csv', delimiter=',')
    edges_df = pd.read_csv(data_dir + 'edges.csv', delimiter=',')
    edges_np = edges_df.to_numpy()
    N = len(nodes_df)
    with open(input_route, mode='r', newline='') as file:
        num_lines = sum(1 for line in file)

    # * Collect nodes positions (x, y are in columns 1 and 2)
    # nodes.csv format: osmid,x,y,ref,highway,index
    nodes_np = nodes_df.to_numpy()
    positions = {}
    for i in range(len(nodes_np)):
        x = float(nodes_np[i, 1])  # x coordinate
        y = float(nodes_np[i, 2])  # y coordinate
        positions[i] = np.array([x, y])

    # * Filter nodes
    nodes_dict = {}
    reverse_node_dict = []
    node_num = 0
    route_index = 0
    with open(input_route, mode='r', newline='') as file:
        for line in tqdm(file, total=num_lines, desc="Filter nodes"):
            line = line.strip()
            if not line:
                continue
            # Each line is comma-separated edge IDs
            route = line.split(',')
            for edge_index_str in route:
                edge_index_str = edge_index_str.strip()
                if not edge_index_str:
                    continue
                edge_index = int(edge_index_str)
                if edge_index >= len(edges_np):
                    continue
                edge_row = edges_np[edge_index]
                u_index = int(edge_row[-2])
                v_index = int(edge_row[-1])
                if u_index not in nodes_dict.keys():
                    nodes_dict[u_index] = node_num
                    reverse_node_dict.append(u_index)
                    node_num += 1
                if v_index not in nodes_dict.keys():
                    nodes_dict[v_index] = node_num
                    reverse_node_dict.append(v_index)
                    node_num += 1
            route_index += 1
    print(f'Node num: {node_num}')

    # * Build weighted graph
    route_index = 0
    edges_dict = {}
    false_edges_dict = {}
    with open(input_route, mode='r', newline='') as file:
        for line in tqdm(file, total=num_lines, desc="Building weighted adj"):
            line = line.strip()
            if not line:
                continue
            route = line.split(',')
            for edge_index_str in route:
                edge_index_str = edge_index_str.strip()
                if not edge_index_str:
                    continue
                edge_index = int(edge_index_str)
                if edge_index >= len(edges_np):
                    continue
                edge_row = edges_np[edge_index]
                u_index_orig = int(edge_row[-2])
                v_index_orig = int(edge_row[-1])
                if u_index_orig not in nodes_dict or v_index_orig not in nodes_dict:
                    continue
                u_index = nodes_dict[u_index_orig]
                v_index = nodes_dict[v_index_orig]
                if (u_index, v_index) not in edges_dict and (v_index, u_index) not in edges_dict:
                    edges_dict[(u_index, v_index)] = 1
                elif (u_index, v_index) in edges_dict:
                    edges_dict[(u_index, v_index)] += 1
                elif (v_index, u_index) in edges_dict:
                    edges_dict[(v_index, u_index)] += 1
                else:
                    print('Error')
                # False
                if (u_index, v_index) not in false_edges_dict and (v_index, u_index) not in false_edges_dict:
                    false_edges_dict[(u_index, v_index)] = 1
                elif (u_index, v_index) in false_edges_dict:
                    false_edges_dict[(u_index, v_index)] += 1
                elif (v_index, u_index) in false_edges_dict:
                    false_edges_dict[(v_index, u_index)] += 1
            route_index += 1

    # * Do Heuristic Clustering
    edges_values = list(false_edges_dict.keys())
    weights_values = list(false_edges_dict.values())
    edge_and_weight = [[edges_values[i][0], edges_values[i][1],weights_values[i]] for i in range(len(edges_values))]
    reversed_edge_and_weight = [[edges_values[i][1], edges_values[i][0],weights_values[i]] for i in range(len(edges_values))]
    sorted_edge_and_weight = sorted(edge_and_weight + reversed_edge_and_weight, key=lambda x: x[0])
    
    xadj = []
    adjncy = []
    eweights = []
    current_index = 0
    last_u = -1
    for edge_and_weight in sorted_edge_and_weight:
        u = edge_and_weight[0]
        v = edge_and_weight[1]
        weight = edge_and_weight[2]
        if u != last_u:
            for i in range(last_u+1, u+1):
                xadj.append(current_index)
            last_u = u
        adjncy.append(v)
        eweights.append(weight)            
        current_index+=1
    xadj.append(current_index)

    print('Start Heuristic Clustering')
    n_cuts, hc_membership = pymetis.part_graph(cluster_num, xadj=xadj, adjncy=adjncy, eweights=eweights)
    hc_cluster_centers = calculate_cluster_center(positions=positions, reverse_node_dict=reverse_node_dict, membership=hc_membership, num_class=cluster_num)
    hc_full_membership = get_full_membership(positions=positions, nodes_dict=nodes_dict, membership=hc_membership, cluster_centers=hc_cluster_centers)
    print(f'Heuristic Clustering clusters num: {len(set(hc_membership))}')
    np.savetxt(f"{output_dir}/txt/hc_{cluster_num}clusters.txt", hc_full_membership, fmt="%d")
    print('Heuristic Clustering finished')
    
    # * Do Community Detection
    print('Start Community Detection')
    edges = [list(edge) for edge in false_edges_dict.keys()]
    graph = igraph.Graph(n=node_num, edges=edges, edge_attrs={'weight': list(false_edges_dict.values())})
    cd_cluster = graph.community_multilevel(weights='weight')
    cd_membership = cd_cluster.membership
    cd_cluster_num = len(set(cd_membership))
    cd_cluster_centers = calculate_cluster_center(positions=positions, reverse_node_dict=reverse_node_dict, membership=cd_membership, num_class=cd_cluster_num)
    kmeans = cluster.KMeans(n_clusters=cluster_num, random_state=0, n_init='auto').fit(cd_cluster_centers).labels_
    cd_membership = [kmeans[membership] for membership in cd_membership]
    cd_cluster_centers = calculate_cluster_center(positions=positions, reverse_node_dict=reverse_node_dict, membership=cd_membership, num_class=cluster_num)
    cd_full_membership = get_full_membership(positions=positions, nodes_dict=nodes_dict, membership=cd_membership, cluster_centers=cd_cluster_centers)
    print(f'Community Detection clusters num: {len(set(cd_membership))}')
    np.savetxt(f'{output_dir}/txt/cd_{cluster_num}clusters.txt', cd_full_membership, fmt="%d")
    print('Community Detection finished')
    
    # * Visualization
    hc_plot_colors = {key: get_color(value) for key, value in enumerate(hc_full_membership)}
    graph_networkx = nx.Graph()
    fig, ax = plt.subplots()
    graph_networkx.add_nodes_from([i for i in range(N)])
    graph_networkx.add_edges_from(edges)
    nx.draw_networkx_nodes(graph_networkx, positions, node_size=3, node_color=[hc_plot_colors[node] for node in graph_networkx.nodes], ax=ax)
    # nx.draw_networkx_edges(graph_networkx, positions, width=0.01, edge_color='grey', ax=ax)
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Voyager)
    plt.axis('off')
    plt.savefig(f'{output_dir}/diagram/hc_{cluster_num}clusters.png', dpi=2000)
    plt.clf()

    cd_plot_colors = {key: get_color(value) for key, value in enumerate(cd_full_membership)}
    fig, ax = plt.subplots()
    nx.draw_networkx_nodes(graph_networkx, positions, node_size=3, node_color=[cd_plot_colors[node] for node in graph_networkx.nodes], ax=ax)
    # nx.draw_networkx_edges(graph_networkx, positions, width=0.01, edge_color='grey', ax=ax)
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.CartoDB.Voyager)
    plt.axis('off')
    plt.savefig(f'{output_dir}/diagram/cd_{cluster_num}clusters.png', dpi=2000, )
    plt.clf()

    hc_matrix = [[0 for _ in range(cluster_num)] for _ in range(cluster_num)]
    cd_matrix = [[0 for _ in range(cluster_num)] for _ in range(cluster_num)]
    
    passed_edges_dict = {}
    for edge_row in edges_np:
        u_index = int(edge_row[-2])
        v_index = int(edge_row[-1])

        hc_u_cluster = hc_full_membership[u_index]
        hc_v_cluster = hc_full_membership[v_index]
        cd_u_cluster = cd_full_membership[u_index]
        cd_v_cluster = cd_full_membership[v_index]

        if u_index not in nodes_dict.keys() or v_index not in nodes_dict.keys():
            continue
        u_index = nodes_dict[u_index]
        v_index = nodes_dict[v_index]
        if (u_index, v_index) not in passed_edges_dict:
            if (u_index, v_index) in edges_dict:
                weight = edges_dict[(u_index, v_index)]
            elif (v_index, u_index) in edges_dict:
                weight = edges_dict[(v_index, u_index)]
            else:
                weight = 0
            passed_edges_dict[(u_index, v_index)] = True
            passed_edges_dict[(v_index, u_index)] = True
            hc_matrix[hc_u_cluster][hc_v_cluster] += weight
            cd_matrix[cd_u_cluster][cd_v_cluster] += weight
            if hc_u_cluster != hc_v_cluster:
                hc_matrix[hc_v_cluster][hc_u_cluster] += weight
            if cd_u_cluster != cd_v_cluster:
                cd_matrix[cd_v_cluster][cd_u_cluster] += weight

    hc_matrix = np.array(hc_matrix)
    cd_matrix = np.array(cd_matrix)

    plt.figure(dpi=200, figsize=(16, 16))
    plt.imshow(hc_matrix, cmap='viridis')
    plt.colorbar()
    for i in range(hc_matrix.shape[0]):
        for j in range(hc_matrix.shape[1]):
            if hc_matrix[i, j] > hc_matrix.max() / 2:
                text_color = 'black'
            else:
                text_color = 'white'
            if hc_matrix[i, j] > 1000:
                plt.text(j, i, "{:.2e}".format(hc_matrix[i, j]), ha='center', va='center', color=text_color)
            else:
                plt.text(j, i, str(hc_matrix[i, j]), ha='center', va='center', color=text_color)
    plt.xticks([i for i in range(cluster_num)], [i for i in range(cluster_num)])
    plt.yticks([i for i in range(cluster_num)], [i for i in range(cluster_num)])
    plt.savefig(f'{output_dir}/connectivity/hc_{cluster_num}clusters.png', dpi=200)
    plt.clf()

    plt.figure(dpi=200, figsize=(16, 16))
    plt.imshow(cd_matrix, cmap='viridis')
    plt.colorbar()
    for i in range(cd_matrix.shape[0]):
        for j in range(cd_matrix.shape[1]):
            if cd_matrix[i, j] > cd_matrix.max() / 2:
                text_color = 'black'
            else:
                text_color = 'white'
            if cd_matrix[i, j] > 1000:
                plt.text(j, i, "{:.2e}".format(cd_matrix[i, j]), ha='center', va='center', color=text_color)
            else:
                plt.text(j, i, str(cd_matrix[i, j]), ha='center', va='center', color=text_color)
    plt.xticks([i for i in range(cluster_num)], [i for i in range(cluster_num)])
    plt.yticks([i for i in range(cluster_num)], [i for i in range(cluster_num)])
    plt.savefig(f'{output_dir}/connectivity/cd_{cluster_num}clusters.png', dpi=200)
    plt.clf()

if __name__ == "__main__":
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(os.path.join(output_dir, 'txt')):
        os.makedirs(os.path.join(output_dir, 'txt'))
    if not os.path.exists(os.path.join(output_dir, 'diagram')):
        os.makedirs(os.path.join(output_dir, 'diagram'))
    if not os.path.exists(os.path.join(output_dir, 'connectivity')):
        os.makedirs(os.path.join(output_dir, 'connectivity'))
    # * Run two algorithms
    for cluster_num in cluster_nums:
        run_two_algorithms(cluster_num)
    # * All done
    print('All done')

