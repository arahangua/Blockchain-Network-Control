import os,sys
import numpy as np
import pandas as pd
import networkit as nk
import networkx as nx

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')    
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 600

from typing import Union 

from sklearn import preprocessing
'''
Networkit is used for generating / partitioning the graph
Netowrkx is used for more specialized downstream computation/task on these partitioned graphs
'''



class BNC():
    def __init__(self, all_addrs:np.array, token_df:pd.DataFrame, bin_size=7200):
        self.uniq_addrs = np.unique(all_addrs)
        self.get_timewindows(token_df)
        self.set_binnings(bin_size = bin_size)

        print(f'number of unique addrs : {len(self.uniq_addrs)}')
        print('BNC class initialized')

    def get_timewindows(self, token_df):
        self.time_init = True
        self.min_bl = token_df['blockNumber'].min()
        self.max_bl = token_df['blockNumber'].max()
        self.tot_dur = self.max_bl - self.min_bl

        print(f'analyzing block range : {self.min_bl} - {self.max_bl}, duration of {self.tot_dur} blocks')

    def set_binnings(self, bin_size): # roughly 7200 blocks worth 1 day (a bit more than one day)
        self.bin_size = bin_size
        print(f'binning window is set to {self.bin_size} blocks')

    def set_pricetable(self, pricetable):
        self.pricetable = pricetable
        print(f'price table added : {list(self.pricetable.keys())}')

    def lookup_compute_price(self, token_df, value_col = 'value', price_key='symbol'):
        if not hasattr(self, 'pricetable'):
            raise Exception('you need to run set_pricetable() to first set price table.')

        tokens = list(self.pricetable.keys())
        for tk in tokens:
            subset = token_df[token_df[price_key]==tk]
            token_df.loc[token_df[price_key]==tk,value_col] = subset[value_col].apply(lambda x: x* self.pricetable[tk])
        
        return token_df
        



    def get_relevant_transfers(self, transfer_df):
        '''
        transfer_df : should be a dataFrame with 'from', 'to', 'value' columns 
        
        '''
        init_rows = len(transfer_df)
        transfer_df = transfer_df[(transfer_df['from'].isin(self.uniq_addrs) & transfer_df['to'].isin(self.uniq_addrs))]
        transfer_df = self.omit_special_addrs(transfer_df)
        after_rows = len(transfer_df)
        print(f'subsetting done : from {init_rows} rows -> {after_rows} relevant rows')
        return transfer_df
    

    def omit_special_addrs(self, data_df, special_addrs = None):
        # take out edge cases 
        if(special_addrs is None):
            special_addrs = ['0x000000000000000000000000000000000000dead', '0x0000000000000000000000000000000000000000']

        for addr in special_addrs:
            data_df = data_df[data_df['from']!=addr]
            data_df = data_df[data_df['to']!=addr]
        
        return data_df

    def compute_edge_weights(self, data_df, agg_methods:list = ['sum']):
        # get edge weights
        grouped = data_df.groupby(['from', 'to']).agg(agg_methods).reset_index()
        self.grouped = grouped
        return grouped
        
    def assign_edge_indices(self, grouped):
        grouped['from_idx'] = grouped['from'].apply(lambda x: list(self.uniq_addrs).index(x)) # there shouldn't be any error
        grouped['to_idx'] = grouped['to'].apply(lambda x: list(self.uniq_addrs).index(x)) # there shouldn't be any error
        self.grouped=grouped
        return grouped

    def generate_dir_weighted_graph(self, grouped, agg_method='sum'):
        # Create an empty directed, weighted graph
        G = nk.graph.Graph(weighted=True, directed=True)

        # Add nodes; note that node ids in NetworKit are 0-based integers and it doesn't take string as identifiers
        for ii in np.arange(len(self.uniq_addrs)):
            G.addNode()

        # Add directed weighted edges: G.addEdge(source, target, weight)
        for ii,row in grouped.iterrows():
            G.addEdge(row['from_idx'], row['to_idx'], row['value'][agg_method]) # Edge from node 0 to 1 with weight 2.0

        return G

    def remove_self_loop(self, G):
        return G.removeSelfLoops()

    def convert_to_und(self, G):
        return nk.graphtools.toUndirected(G)
    
    def convert_to_unweighted(self, G):
        return nk.graphtools.toUnweighted(G)
    
    def convert_to_nxgraph(self, G):
        return nk.nxadapter.nk2nx(G)
    
    def convert_to_nkgraph(self, nxgraph):
        return nk.nxadapter.nx2nk(nxgraph, weightAttr=None)
    

    # graph decomposition 
    def get_scc(self, G): 
        comps = nk.components.StronglyConnectedComponents(G).run() # considering the directionality we get the largest component in which all vertices are reachable.
        comp_members = comps.getComponents()
        comp_sizes = comps.getComponentSizes()
        result= {}
        result['members'] = comp_members
        result['sizes'] = comp_sizes
        return result
    
    def get_largest_comp(self, G):
        scc = self.get_scc(G)
        comp_sizes = list(scc['sizes'].values())
        
        # sort it
        comp_sorted_idx = np.argsort(comp_sizes)
        largest_comp_size = comp_sizes[comp_sorted_idx[-1]]

        # get the largest comp
        largest_idx = list(scc['sizes'].keys())[comp_sorted_idx[-1]]
        largest_node_ids = scc['members'][largest_idx]
                
        # subset to generate a new graph (nk graph)
        uniq_addrs_lc = self.uniq_addrs[largest_node_ids]

        print(f'largest component size : {largest_comp_size}, underlying graph size was : {len(self.uniq_addrs)}')

        result={}
        result['lc_size'] = largest_comp_size
        result['lc_uniq_addrs'] = uniq_addrs_lc

        return result

    def get_degrees(self, G):
        # make sure to keep G in the right format for the downstream purpose
        nx_graph = nk.nxadapter.nk2nx(G)

        return nx_graph.degree

#####################################################


    # File I/O
    def write_to_graphml(self, G, fname, node_attr_dict=None, edge_attr_dict=None):
        writer = nk.graphio.GraphMLWriter()
        writer.write(G, fname, nodeAttributes=node_attr_dict, edgeAttributes=edge_attr_dict)
        print(f'graphml file: {fname} successfully generated')

    def read_graphml(self, graph_ml_fname):
        reader = nk.graphio.GraphMLReader()
        G = reader.read(graph_ml_fname)
        return G
    


######################################################
    # plotting 
    def plot_component_size_dist(self, scc_result):
        comp_len = list(scc_result['sizes'].values())
        fig = plt.figure()
        ax = fig.add_subplot()
        sns.violinplot(ax=ax,data=comp_len, color='gray', saturation=0.1)
        # sns.catplot(ax=ax,data=comp_len, color='gray', saturation=0.1)
        sns.stripplot(ax=ax,data=comp_len, color='darkblue', alpha=0.5)
        plt.yscale('log')
        ax.set_xlabel('Strongly Connected Components')
        ax.set_ylabel('Component Size')
        ax.set_xticklabels([])

    def general_profile(self,G, config_setting='minimal'):
        # general profiling
        import warnings
        warnings.filterwarnings('ignore')
        config = nk.profiling.Config.createConfig(preset=config_setting)
        nk.profiling.Profile.create(G, config=config).show() # need to use subsets otherwise takes too long



#######################################################
    # some calculations 
    def get_knee_point(self, list_1d: Union[list, np.array], return_plot=False): # any 1d data: list or np.array
        # scale it right 
        standardized = preprocessing.scale(list_1d)
        sorted = np.sort(standardized)
        # get x,y coord for max, min
        max_x = len(sorted)
        min_x = 1

        max_y = sorted[-1]
        min_y = sorted[0]

        beta = (max_y-min_y) / (max_x - min_x)
        ortho_beta = -1/beta

        # needs to compute all points as the curve is not actually smooth
        heights = []
        for_indices = np.arange(len(sorted))[::-1] # reverse it to match the sorting scheme
        for x in for_indices:
            intercept = sorted[x] - (ortho_beta*(x+1))

            #intersection point 
            inter_x = intercept/(beta - ortho_beta) 
            inter_y = beta*inter_x

            on_curve_p = np.array((x+1, sorted[x]))
            on_line_p = np.array((inter_x, inter_y))

            height = np.linalg.norm(on_curve_p - on_line_p)
            heights.append(height)
        
        knee_point = np.argmax(heights)        
        print(f'found a maximum at : {knee_point} (index in descending order), {sorted[-(knee_point+1)]}')
        if(return_plot):
            plt.plot(heights)
            plt.xlabel('nodes')
            plt.ylabel('length of perpendicular projection')

        return knee_point
######################################################
    def get_maximal_flow(self, G, indices_list, knee_point):
        """
        G: networkit graph (directed, weighted)
        indice_list: list of node_ids (in a networkit graph) sorted in descending order in respect to the metric of interest
        knee_point: index of knee_point from the above indice_list (not id)
        """
        # assign nodes to supersource / supersink
        attach_to_sink = indices_list[:knee_point]
        source_nodes = indices_list[knee_point:]

        # supersource 
        G.addNode()
        nodes_num = G.numberOfNodes() # index of a supersource node
        for ii in source_nodes: 
            G.addEdge(nodes_num-1,ii, 1) # artificial edges have weights of 1
        s_source_idx = nodes_num -1 


        # supersink
        G.addNode()
        nodes_num = G.numberOfNodes() # index of a supersink node
        for ii in attach_to_sink: # supersource is also left out
            G.addEdge(ii, nodes_num-1, 1)
        s_sink_idx = nodes_num -1 

        # use networkx max flow methods
        nx_graph = nk.nxadapter.nk2nx(G) # 'weight' also gets exported
        result = {}
        result['source_nodes'] = source_nodes
        result['max_flow'] =  nx.maximum_flow(nx_graph, s_source_idx, s_sink_idx, capacity='weight')
        
            
        return result


        # networkit native ver (not used)
        # # apply maximum flow (for networkit)
        # G.indexEdges()
        # max_flow = nk.flow.EdmondsKarp(G, s_source_idx, s_sink_idx).run()
        # flow_vec = max_flow.getFlowVector()




    def get_unmatched_nodes_count(self, G):
        """
        G: networkit graph (unweighted, directed (already should be scc))
        returns the number of unmatched nodes
        """
        nx_graph=nk.nxadapter.nk2nx(G)

        # convert it to undirected
        und_dx_graph = nx.to_undirected(nx_graph)

        # get components
        nx.components.connected_components(und_dx_graph)

        # compute maximum matching 
        matched_edges = sorted(nx.maximal_matching(und_dx_graph))

        # count matched nodes (receiving end of edges)
        matched_edges = np.array(matched_edges)
        uniq_matched_nodes = np.unique(matched_edges[:,1])

        # total number of nodes
        tot_nodes = len(list(und_dx_graph.nodes()))

        # number of unmatched nodes ()
        unmatched_nodes = tot_nodes - len(uniq_matched_nodes)
        print(f'number of unmatched nodes: {unmatched_nodes}')

        return unmatched_nodes
    
    def get_unmatched_nodes_count_ER(self, G, iter=30):        
        """
        G: networkit graph (unweighted, directed (already should be scc))
        returns the number of unmatched nodes for each surrogate instance (ER)
        """
        nx_graph=nk.nxadapter.nk2nx(G)
        # convert it to undirected
        und_dx_graph = nx.to_undirected(nx_graph)
        # total number of nodes
        tot_nodes = len(list(und_dx_graph.nodes()))
        
        density = nx.density(und_dx_graph)
        print(f'current density of the undirected graph : {density}')
        # generate ER surrogate
        
        unmatched_list = []
        for _ in np.arange(iter):
            erg = nk.generators.ErdosRenyiGenerator(tot_nodes, density, directed=False)
            # Run algorithm
            ergG = erg.generate()
            # verify creation
            print(ergG.numberOfNodes(), ergG.numberOfEdges())
            
            ergG_nx=nk.nxadapter.nk2nx(ergG)
            # check maximum matching 
            # compute maximum matching 
            matched_edges = sorted(nx.maximal_matching(ergG_nx))
            # count matched nodes (receiving end of edges)
            matched_edges = np.array(matched_edges)
            uniq_matched_nodes = np.unique(matched_edges[:,1])
            
            num_unmatched = tot_nodes - len(uniq_matched_nodes)
            print(f'number of unmatched nodes: {num_unmatched}')
            unmatched_list.append(num_unmatched)

        return unmatched_list
    
    def get_hits_nodes(self, G, max_iter=None, tol=1e-08, nstart=None, normalized=True):
        """
        G : networkit graph (directed, weighted)
        
        """
        nx_graph=nk.nxadapter.nk2nx(G)
        hubs,authorities = nx.hits(nx_graph, max_iter=None, tol=1e-08, nstart=None, normalized=True)

        return hubs, authorities
    
    def get_influence(self, max_flow_result):
        inf_df = pd.DataFrame()
        res= {}
        path_result = max_flow_result['max_flow'][1]
        real_nodes_num = len(self.uniq_addrs)
        for k,v in path_result.items():
            if(k<real_nodes_num):
                res['addr'] = self.uniq_addrs[k]
                influence = np.sum(list(v.values()))
                res['influence'] = influence

                res_df = pd.DataFrame(res, index=[0])
                inf_df = pd.concat([inf_df, res_df])

        inf_df.reset_index(drop=True, inplace=True)
        inf_df= inf_df.sort_values('influence', ascending=False)

        return inf_df