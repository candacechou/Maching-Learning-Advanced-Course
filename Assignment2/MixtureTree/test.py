import argparse
import numpy as np
import matplotlib.pyplot as plt
from math import isnan
from itertools import combinations
import Kruskal_v2 as Kv
import sys
from Tree import Node, Tree, TreeMixture
import dendropy
def save_results(loglikelihood, topology_array, theta_array, filename):
    """ This function saves the log-likelihood vs iteration values,
        the final tree structure and theta array to corresponding numpy arrays. """

    likelihood_filename = filename + "_em_loglikelihood.npy"
    topology_array_filename = filename + "_em_topology.npy"
    theta_array_filename = filename + "_em_theta.npy"
    print("Saving log-likelihood to ", likelihood_filename, ", topology_array to: ", topology_array_filename,
          ", theta_array to: ", theta_array_filename, "...")
    np.save(likelihood_filename, loglikelihood)
    np.save(topology_array_filename, topology_array)
    np.save(theta_array_filename, theta_array)

def calculate_likelihood(sample,topology,theta):
    """
    :param sample : a list of sample
    :param topology : a list of its parent
    :param theta    : CPD

    """
    num_node = len(sample)
    resp     = 1
    #print(len(theta))
    for i in range(num_node):

        ### root
        if isnan(topology[i]):
            resp = resp * theta[0][sample[i]]
        else:
            parent = topology[i]
            psam = sample[int(parent)]
            csam = sample[i]
            CPD  = theta[i]
            resp = resp * CPD[psam][csam]

    return resp

def Compute_qkab(rk,node1,node2,a,b):
    if a == 0 :
        find1 = np.where(node1 == 0)[0]
    else :
        find1 = np.where(node1 == 1)[0]
    if b == 0 :
        find2 = np.where(node2 == 0)[0]
    else :
        find2 = np.where(node2 == 1)[0]
    index = list(set(find1).intersection(set(find2)))
    weight = np.sum(rk[(index)]) / np.sum(rk)
    return weight

def Compute_qka(rk,node1,a):
    if a == 0:
        find1 = np.where(node1 == 0)[0]
    else:
        find1 = np.where(node1 == 1)[0]

    weight = np.sum(rk[find1]) / np.sum(rk)

    return weight

def ComputeWeight(node,r,k,sample):
    weight = 0
    node1 = sample[:,node[0]].reshape((sample.shape[0],1))
    node2 = sample[:,node[1]].reshape((sample.shape[0],1))
    rk    = r[:,k].reshape((sample.shape[0],1))
    for a in range(2):
        for b in range(2):
            qkab = Compute_qkab(rk, node1, node2, a, b)
            qka  = Compute_qka(rk, node1, a)
            qkb  = Compute_qka(rk, node2, b)
            if qka == 0 or qkb == 0 or qkab == 0:
                weight += 0
            else:
                weight += qkab * np.log(qkab/(qka * qkb))
    return weight

def generate_graph(vertex,edge):

    graph = {
        'vertices': [i for i in range(vertex)],
        'edges': {i for i in edge}

    }
    return graph
def Find_Sub_topology(result,parent,tree):
    o = []
    new_result = []
    for (u,v) in result :
        if u == parent:
            if v not in o:
                o.append(v)
            tree[v] = parent
        elif v == parent :
            if u not in o :
                o.append(u)
            tree[u] = parent
        else:
            new_result.append((u,v))
    if len(o) == 0 :
        return new_result,tree
    else :
        for sub_parent in o :
            new_result, tree = Find_Sub_topology(new_result,sub_parent,tree)
        return new_result, tree
def Find_New_topology(result,num_nodes):
    topology = [np.nan] * num_nodes
    new_result = []
    o = []
    ### find first children of 0
    for (u,v,w) in result:

        if u == 0 :
            if v not in o :
                o.append(v)
            topology[v] = 0
        elif v == 0 :
            if u not in o :
                o.append(u)
            topology[u] = 0
        else :
            new_result.append((u,v))



    for subroot in o :
        new_result ,topology = Find_Sub_topology(new_result, subroot, topology)

    return topology
def Compute_theta(r,k,topology,num_node,samples):
    """

    :param r: responsibility
    :param k: tree number
    :param topology: topology
    ;param num_node : number of nodes
    ;param samples : samples
    :return: theta : list of array of CPD
    """
    theta = []
    ### the root

    rk   = r[:,k]
    root = [Compute_qka(rk,samples[:,0],0),Compute_qka(rk,samples[:,0],1)]
    root = root/np.sum(root)
    theta.append(root)
    for i in range(1, num_node):
        sub_theta = np.zeros((2,2))
        parent = topology[i]
        for a in range(2): ## parent
            for b in range(2): ## child
                given_p = np.where(samples[:,parent] == a)[0]
                given_c = np.where(samples[:, i] == b)[0]
                index = list(set(given_p).intersection(set(given_c)))
                sub_theta[a][b] = np.sum(rk[index])/np.sum(rk[given_p]) + sys.float_info.epsilon

        sub_theta = sub_theta / np.tile(np.sum(sub_theta, axis = 1).reshape((2, 1)),2)
        ### to fit the data structure from TA = ...

        theta.append([sub_theta[0,:],sub_theta[1,:]])

    return theta
def Compute_likelihood_from_real_tree(mixtree, samples):
    likelihood = 0
    pi = mixtree.pi
    n_samples = samples.shape[0]
    num_clusters = len(pi)
    r = np.zeros((n_samples, num_clusters))
    for n in range(n_samples):
        for c in range(num_clusters):
            topology = mixtree.clusters[c].get_topology_array()
            theta    = mixtree.clusters[c].get_theta_array()
            r[n, c] = calculate_likelihood(samples[n], topology, theta) + sys.float_info.epsilon
    likelihood = np.sum(np.log(np.sum(r * pi, axis=1)))
    return likelihood

def em_algorithm(seed_val, samples, num_clusters, max_num_iter=100):
    """
    This function is for the EM algorithm.
    :param seed_val: Seed value for reproducibility. Type: int
    :param samples: Observed x values. Type: numpy array. Dimensions: (num_samples, num_nodes)
    :param num_clusters: Number of clusters. Type: int
    :param max_num_iter: Maximum number of EM iterations. Type: int
    :return: loglikelihood: Array of log-likelihood of each EM iteration. Type: numpy array.
                Dimensions: (num_iterations, ) Note: num_iterations does not have to be equal to max_num_iter.
    :return: topology_list: A list of tree topologies. Type: numpy array. Dimensions: (num_clusters, num_nodes)
    :return: theta_list: A list of tree CPDs. Type: numpy array. Dimensions: (num_clusters, num_nodes, 2)

    You can change the function signature and add new parameters. Add them as parameters with some default values.
    i.e.
    Function template: def em_algorithm(seed_val, samples, k, max_num_iter=10):
    You can change it to: def em_algorithm(seed_val, samples, k, max_num_iter=10, new_param_1=[], new_param_2=123):
    """

    # Set the seed
    np.random.seed(seed_val)
    # TODO: Implement EM algorithm here.
    # Start: Example Code Segment. Delete this segment completely before you implement the algorithm.
    #### randomly create trees
    print("Running EM algorithm...")
    loglikelihood = []
    # for iter_ in range(max_num_iter):
    #     loglikelihood.append(np.log((1 + iter_) / max_num_iter))
    from Tree import TreeMixture
    sieving = 20
    bestlikelihood = 0
    #tm = TreeMixture(num_clusters=num_clusters, num_nodes = samples.shape[1])
    for i in range(sieving):
        tm = TreeMixture(num_clusters=num_clusters, num_nodes = samples.shape[1])
        tm.simulate_pi(seed_val=seed_val)
        tm.simulate_trees(seed_val=seed_val)
        tm.sample_mixtures(num_samples=samples.shape[0], seed_val = seed_val)
        like = Compute_likelihood_from_real_tree(tm, samples)
        if (np.exp(like) > bestlikelihood) or i == 0:
            bestlikelihood = like
            topology_list = []
            theta_list = []

            for k in range(num_clusters):
                topology_list.append(tm.clusters[k].get_topology_array())
                theta_list.append(tm.clusters[k].get_theta_array())



    topology_list = np.array(topology_list)
    theta_list = np.array(theta_list)
    pi         = np.ones((1,num_clusters))
    pi         = pi / np.sum(pi)

    #### start do EM algorithm
    n_samples = samples.shape[0]
    ### responsibility
    ### 1. calculate p(sample|tk,thetak)
    for i in range(max_num_iter):
        r = np.zeros((n_samples,num_clusters))
        for n in range(n_samples):
            for c in range(num_clusters):
                r[n,c] =  calculate_likelihood(samples[n],topology_list[c],theta_list[c]) + sys.float_info.epsilon
            #r[n,:] = r[n,:]/ np.sum(r[n,:])

    ### 2. responsibility
        loglikelihood.append(np.sum(np.log(np.sum(r * pi, axis=1))))
        r = pi * (r / np.tile(np.sum(r * pi, axis = 1).reshape((n_samples, 1)),num_clusters))
        #print(r)

    ### 3. pi
        pi = np.mean(r, axis=0)
        #pi = pi / np.sum(pi)
    ### 4. compute weight
        NoN = list(combinations([i for i in range(samples.shape[1])],2))
        topology_list = []
        theta_list = []
        for k in range(num_clusters):
            edges = []
            for e in NoN:
                # edge = (node1 , node2)
                nodeWedge = ComputeWeight(e, r, k, samples)
                if nodeWedge != 0:
                    edges.append((e[0], e[1], nodeWedge))
            graphs = generate_graph(samples.shape[1],edges)
            result = Kv.maximum_spanning_tree(graphs)
            new_topology = Find_New_topology(result, samples.shape[1])
            topology_list.append(new_topology)
            new_theta = Compute_theta(r , k , new_topology , samples.shape[1],samples)
            theta_list.append(new_theta)

        theta_list = np.array(theta_list)
        topology_list = np.array(topology_list)
        if abs(loglikelihood[i] - loglikelihood[i-1]) < sys.float_info.epsilon and i != 0:
            loglikelihood = np.array(loglikelihood)
            return loglikelihood, topology_list, theta_list


    return loglikelihood, topology_list, theta_list


def main():
    # Code to process command line arguments
    # x = [i for i in range(2,16)]
    # y = [-80.568,-63.394,-59.915,-60.438,-59.91,-59.9146,-59.9156,-59.9146,-59.9146,-59.9146,-59.91464,-59.91462,-59.914,-59.915]
    # plt.figure()
    # plt.scatter(x, y )
    # plt.xlabel('number of clusters')
    # plt.ylabel('log-likelihood')
    # plt.title('The likelihood in different number of clusters')
    # plt.show()
    num_clusters = 4
    num_samples  = 20
    num_nodes    = 10
    seed_vals    = 42
    real_Mixture = TreeMixture(num_clusters,num_nodes)
    real_Mixture.simulate_pi(seed_vals)
    real_Mixture.simulate_trees(seed_vals)
    real_Mixture.sample_mixtures(num_samples,seed_vals)

    samples = real_Mixture.samples



    loglikelihood, topology_array, theta_array = em_algorithm(seed_vals, samples, num_clusters = 3, max_num_iter = 200)


    #### generate trees
    Tree_list = []


    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    plt.plot(np.exp(loglikelihood), label='Estimated')
    plt.ylabel("Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.subplot(122)
    plt.plot(loglikelihood, label='Estimated')
    plt.ylabel("Log-Likelihood of Mixture")
    plt.xlabel("Iterations")
    plt.legend(loc=(1.04, 0))
    plt.show()

    print("\n4. Retrieve real results and compare.\n")

    for i in range(3):
        single_tree = Tree()
        single_tree.load_tree_from_direct_arrays(topology_array[i], theta_array[i])
        Tree_list.append(single_tree)

    #### Compare the RF distance:
    tns = dendropy.TaxonNamespace()
    for ntt in range(num_clusters):
        for itt in range(3):
            tt = dendropy.Tree.get(data = real_Mixture.clusters[ntt].newick, schema="newick", taxon_namespace=tns)
            it = dendropy.Tree.get(data = Tree_list[itt].newick, schema="newick", taxon_namespace=tns)
            print("\tRF distance with :",'true tree n=',ntt,'and inferred tree n=',itt,':\t', dendropy.calculate.treecompare.symmetric_difference(tt, it))
    print("\t4.2. Make the likelihood comparison.\n")
     # TODO: Do Likelihood Comparison
    likelihood_real_tree = Compute_likelihood_from_real_tree(real_Mixture, samples)
    print('loglikelihood in real mixtures :' ,likelihood_real_tree,'likelihood: ',np.exp(likelihood_real_tree))
    print('loglikelihood in inferred mixtures :', loglikelihood[-1],'likelihood: ',np.exp(loglikelihood[-1]))

if __name__ == "__main__":
    main()
