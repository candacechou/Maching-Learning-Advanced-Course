""" This file is created as a template for question 2.5 in DD2434 - Assignment 2.

    Please keep the fixed parameters in the function templates as is (in 2_5.py file).
    However if you need, you can add parameters as default parameters.
    i.e.
    Function template: def em_algorithm(seed_val, samples, k, num_iter=10):
    You can change it to: def em_algorithm(seed_val, samples, k, num_iter=10, new_param_1=[], new_param_2=123):

    You can write helper functions however you want.

    You do not have to implement the code for finding a maximum spanning tree from scratch. We provided two different
    implementations of Kruskal's algorithm and modified them to return maximum spanning trees as well as the minimum
    spanning trees. However, it will be beneficial for you to try and implement it. You can also use another
    implementation of maximum spanning tree algorithm, just do not forget to reference the source (both in your code
    and in your report)!

    We also provided an example regarding the Robinson-Foulds metric (see Phylogeny.py).

    If you want, you can use the class structures provided to you (Node, Tree and TreeMixture classes in Tree.py file),
    and modify them as needed. In addition to the sample files given to you, it is very important for you to test your
    algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format.

    Note that the sample files are tab delimited with binary values (0 or 1) in it.
    Each row corresponds to a different sample, ranging from 0, ..., N-1
    and each column corresponds to a vertex from 0, ..., V-1 where vertex 0 is the root.
    Example file format (with 5 samples and 4 nodes):
    1   0   1   0
    1   0   1   0
    1   0   0   0
    0   0   1   1
    0   0   1   1

    Also, I am aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). I wanted to keep the template codes as simple as possible.
    You can change the file names however you want (i.e tmm_1_samples.txt).

    After all, we will test your code with commands like this one:
    %run 2_5.py "data/example_tree_mixture.pkl_samples.txt" "data/example_result" 3 --seed_val 123
    where
    "data/example_tree_mixture.pkl_samples.txt" is the filename of the samples
    "data/example_result" is the base filename of results (i.e data/example_result_em_loglikelihood.npy)
    3 is the number of clusters for EM algorithm
    --seed_val is the seed value for your code, for reproducibility.

    For this assignment, we gave you three different trees
    (q_2_5_tm_10node_20sample_4clusters, q_2_5_tm_10node_50sample_4clusters, q_2_5_tm_20node_20sample_4clusters).
    As the names indicate, the mixtures have 4 clusters with varying number of nodes and samples.
    We want you to run your EM algorithm and compare the real and inferred results in terms of Robinson-Foulds metric
    and the likelihoods.
    """
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
    sieving = 1
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
    parser = argparse.ArgumentParser(description='EM algorithm for likelihood of a tree GM.')
    parser.add_argument('sample_filename', type=str, default = '/Users/candacechou/Desktop/homework/advML/Assignment2/AdvML19-master/2_5/data/q_2_5_tm_10node_20sample_4clusters.pkl_samples.txt',
                        help='Specify the name of the sample file (i.e data/example_samples.txt)')
    parser.add_argument('output_filename', type=str, default = '/Users/candacechou/Desktop/homework/advML/Assignment2/AdvML19-master/2_5/result0',
                        help='Specify the name of the output file (i.e data/example_results.txt)')
    parser.add_argument('num_clusters', type=int,default = 4, help='Specify the number of clusters (i.e 3)')
    parser.add_argument('seed_val', type=int, default= 512, help='Specify the seed value for reproducibility (i.e 42)')
    parser.add_argument('--real_values_filename', type=str, default='',
                        help='Specify the name of the real values file (i.e data/example_tree_mixture.pkl)')
    # You can add more default parameters if you want.

    # print("Hello World!")
    # print("This file demonstrates the flow of function templates of question 2.5.")

    print("\n0. Load the parameters from command line.\n")

    args = parser.parse_args()
    print("\tArguments are: ", args)

    print("\n1. Load samples from txt file.\n")

    samples = np.loadtxt(args.sample_filename, delimiter="\t", dtype=np.int32)
    num_samples, num_nodes = samples.shape
    # print("\tnum_samples: ", num_samples, "\tnum_nodes: ", num_nodes)
    # print("\tSamples: \n", samples)

    print("\n2. Run EM Algorithm.\n")

    loglikelihood, topology_array, theta_array = em_algorithm(args.seed_val, samples, num_clusters=args.num_clusters, max_num_iter= 200)
    print(topology_array)
    print(theta_array)
    print("\n3. Save, print and plot the results.\n")

    save_results(loglikelihood, topology_array, theta_array, args.output_filename)
    #### generate trees
    Tree_list = []

    # for i in range(args.num_clusters):
    #     print("\n\tCluster: ", i)
    #     print("\tTopology: \t", topology_array[i])
    #     print("\tTheta: \t", theta_array[i])

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
    if args.real_values_filename != "":
        print("\tComparing the results with real values...")

        print("\t4.1. Make the Robinson-Foulds distance analysis.\n")
        # TODO: Do RF Comparison
        #### tree tree
        true_trees = TreeMixture(0,0)
        true_trees.load_mixture(args.real_values_filename)
        #### Inferred trees
    else:
        tree_real_values_filename = args.sample_filename.split('.')
        tree_file_name = tree_real_values_filename[0] +'.pkl'
        true_trees = TreeMixture(0, 0)
        true_trees.load_mixture(tree_file_name)
    for i in range(args.num_clusters):
        single_tree = Tree()
        single_tree.load_tree_from_direct_arrays(topology_array[i], theta_array[i])
        Tree_list.append(single_tree)

    #### Compare the RF distance:
    tns = dendropy.TaxonNamespace()
    for ntt in range(len(true_trees.clusters)):
        for itt in range(args.num_clusters):
            # print('true tree : ',ntt)
            # true_trees.clusters[ntt].print_topology()
            # print('inferred tree : ',itt)
            # Tree_list[itt].print_topology()
            tt = dendropy.Tree.get(data=true_trees.clusters[ntt].newick, schema="newick", taxon_namespace=tns)
            it = dendropy.Tree.get(data=Tree_list[itt].newick, schema="newick", taxon_namespace=tns)
            print("\tRF distance with :",'true tree n=',ntt,'and inferred tree n=',itt,':\t', dendropy.calculate.treecompare.symmetric_difference(tt, it))
    print("\t4.2. Make the likelihood comparison.\n")
     # TODO: Do Likelihood Comparison
    likelihood_real_tree = Compute_likelihood_from_real_tree(true_trees, samples)
    print('loglikelihood in real mixtures :' ,likelihood_real_tree,'likelihood: ',np.exp(likelihood_real_tree))
    print('loglikelihood in inferred mixtures :', loglikelihood[-1],'likelihood: ',np.exp(loglikelihood[-1]))

if __name__ == "__main__":
    main()
