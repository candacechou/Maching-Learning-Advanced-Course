""" 

    This file is created as the solution template for question 2.3 in DD2434 - Assignment 2.
    
    For test: 
        there are three test cast:
        1. q_2_3_small_tree
        2. q_2_3_medium_tree
        3. q_2_3_large_tree
        
        Each tree have 5 samples 
"""

import numpy as np
from Tree import Tree
from Tree import Node
from math import isnan
import argparse
import os

def calculateLikelihood(tree_topology, theta, beta):
    """
        This function calculates the likelihood of a sample of leaves.
    Input:
        tree_topology: A tree topology.(numpy array(num_nodes, ))
        theta: CPD of the tree. (numpy array(num_nodes, K))
        beta: A list of node assignments.( numpy array(num_nodes, ))
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
                
    Output:
        likelihood: The likelihood of beta. Type: float.

    """
    likelihood = calculate_s(tree_topology,theta,beta,0)
    likelihood = np.dot(theta[0],likelihood)
    return likelihood

def calculate_s(tree, theta, beta, node):
    
    if isnan(beta[node]):
        children = findChildren(tree,node)
        theta1   = theta[children[0]]
        theta2   = theta[children[1]]
        theta1   = np.array(theta1.tolist())
        theta2   = np.array(theta2.tolist())
        s1 = calculate_s(tree,theta,beta,children[0])
        s2 = calculate_s(tree,theta,beta,children[1])
        s1 = np.dot(theta1,s1)
        s2 = np.dot(theta2,s2)
    
        return s1 * s2
            
    else:
        k = theta.shape[1]
        s = np.zeros((k,1))
        s[int(beta[node])] = 1
        return s
        

def findChildren(tree_topology,node):
    children = []
    children = np.argwhere(tree_topology == node)
    children = children.reshape((2,))
    
    return list(children)
  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--int_dir', type = str, default = "data/q2_3_large_tree.pkl",help="the directory of the output tree")
    args = parser.parse_args()
    filename = args.int_dir # "c", "data/q2_3_large_tree.pkl"
    print("1. Load tree data from file and print it.")
    
    if not os.path.exists(filename):
        return ValueError("No such filename.")
    t = Tree()
    t.load_tree(filename)
    t.print_topology()

    print("2. Calculate likelihood of each FILTERED sample")

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("Sample: ", sample_idx)
        sample_likelihood = calculateLikelihood(t.get_topology_array(), t.get_theta_array(), beta)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
