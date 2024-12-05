from model import Network
import initialize
import matplotlib.pyplot as plt ## This module allows us to create plots
import networkx as nx
import numpy as np
from tqdm import tqdm

def generate_data(number_of_agents, number_of_positive_evidence, connectivity_index, global_self_confidence_level):
    '''This function generates the data for the simulation, including:
       1. self confidence level; 
       2. social network; 
       3. structure of understanding; 
       4. initial confidence level on evidence'''
    # generate the data
    initialize.social_network(number_of_agents, connectivity_index)
    initialize.structure_of_understanding(number_of_agents, number_of_positive_evidence)
    initialize.initial_confidence_level_on_evidence(number_of_agents, number_of_positive_evidence)
    # prepare to load them in the simulation
    social_network_filename = "data/social_network.txt"
    initial_confidence_level_on_evidence_data_filename = "data/initial_confidence_level_on_evidence.txt"
    structure_of_understanding_data_filename = "data/structure_of_understandings.txt"
    self_confidence_data = initialize.constant_self_confidence_level(number_of_agents, global_self_confidence_level)
    return [self_confidence_data, social_network_filename,
            structure_of_understanding_data_filename,
            initial_confidence_level_on_evidence_data_filename]


def generate_polarized_data_with_community(number_of_agents, number_of_positive_evidence, connectivity_index, global_self_confidence_level, p):
    '''This function generates the data for the simulation of a polarized population with community structure, including:
       1. self confidence level;
       2. social network;
       3. structure of understanding;
       4. initial confidence level on evidence'''
    # generate the data.
    initialize.social_network_with_communities(number_of_agents, connectivity_index)
    initialize.structure_of_understanding_with_community(number_of_agents, number_of_positive_evidence, p)
    initialize.initial_confidence_level_on_evidence_with_community(number_of_agents, number_of_positive_evidence)
    # prepare to load them in the simulation
    social_network_filename = "data/social_network_with_community.txt"
    initial_confidence_level_on_evidence_data_filename = "data/initial_confidence_level_on_evidence_with_community.txt"
    structure_of_understanding_data_filename = "data/structure_of_understandings_with_community.txt"
    self_confidence_data = initialize.constant_self_confidence_level(number_of_agents, global_self_confidence_level)
    return [self_confidence_data, social_network_filename,
            structure_of_understanding_data_filename,
            initial_confidence_level_on_evidence_data_filename]


def generate_polarized_data(number_of_agents, number_of_positive_evidence, connectivity_index, global_self_confidence_level, p):
    '''This function generates the data for the simulation of a polarized population without community structure, including:
       1. self confidence level;
       2. social network;
       3. structure of understanding;
       4. initial confidence level on evidence'''
    # generate the data.
    initialize.social_network(number_of_agents, connectivity_index)
    initialize.structure_of_understanding_with_community(number_of_agents, number_of_positive_evidence, p)
    initialize.initial_confidence_level_on_evidence_with_community(number_of_agents, number_of_positive_evidence)
    # prepare to load them in the simulation
    social_network_filename = "data/social_network.txt"
    initial_confidence_level_on_evidence_data_filename = "data/initial_confidence_level_on_evidence_with_community.txt"
    structure_of_understanding_data_filename = "data/structure_of_understandings_with_community.txt"
    self_confidence_data = initialize.constant_self_confidence_level(number_of_agents, global_self_confidence_level)
    return [self_confidence_data, social_network_filename,
            structure_of_understanding_data_filename,
            initial_confidence_level_on_evidence_data_filename]


def generate_belief_distribution(data, network_visualization = False, output_folder = "output_1"):
    '''This function generates the belief distribution plot given the data and output folder'''
    BPN = Network(*data)
    pos = nx.spring_layout(BPN.G)  # You can choose other layouts if needed
    # 40 steps usually is enough to reach a stable state
    for i in tqdm(range(40), desc="Generating belief data"):
        believer_certainty = [agent.belief for agent in BPN.schedule.agents]
        BPN.generate_belief_distribution(i, believer_certainty, output_folder)
        if network_visualization:
            BPN.generate_network_visualization(i, pos, believer_certainty, output_folder)
        BPN.step()


def generate_social_pressure_distribution(data, output_folder = "output_1"):
    '''This function calculates the social pressure distribution of the network'''
    BPN = Network(*data)
    # 40 steps usually is enough to reach a stable state
    for i in tqdm(range(40), desc="Generating social pressure data"):
        believer_certainty = [agent.belief for agent in BPN.schedule.agents]
        BPN.generate_social_pressure_distribution(i, believer_certainty, output_folder)
        BPN.step()


def std_versus_population_size_data(number_of_positive_evidence, connectivity_index):
    '''This function generates the data of standard deviation of belief distribution versus population size for different confidence levels'''
    std_bin = [[], [], [], [], []]
    confidence_levels = [0, 0.25, 0.5, 0.75, 1]
    for k in range(5):
        for size in tqdm(range(100, 1500, 10), 
                         desc="Generating standard deviation plot for confidence level = " + str(confidence_levels[k])):
            bin = []
            for i in range(10):
                data = generate_data(size, number_of_positive_evidence, connectivity_index, confidence_levels[k])
                BPN = Network(*data)
                for j in range(50):
                    BPN.step()
                    believer_certainty = [agent.belief for agent in BPN.schedule.agents]
                bin.append(np.std(believer_certainty))
            std_bin[k].append(np.mean(bin))
    std_bin = np.array(std_bin)
    np.savetxt("std_versus_population_size_data.txt", std_bin, delimiter=" ", fmt="%.7f")


def plot_std_versus_population_size():
    '''This function generates the plot of standard deviation of belief distribution versus population size for different confidence levels'''
    plt.figure(figsize=(10, 6))
    confidence_levels = [0, 0.25, 0.5, 0.75, 1]
    std_bin = np.loadtxt("std_versus_population_size_data.txt", delimiter=" ")
    for k in range(5):
        c = confidence_levels[k]
        plt.plot(np.arange(100, 1500, 10), std_bin[k], label= f"global confidence level = {c}")
        # Add title and labels
        plt.title(f'belief distribution at connectivity k = 10')
        plt.xlabel('size of population')
        plt.ylabel('standard deviation of belief distribution')
    plt.legend()
    plt.savefig("std_versus_size.png")


def std_versus_connectivity_data(number_of_positive_evidence, size):
    '''This function generates the data of standard deviation of belief distribution versus connectivity for different confidence levels'''
    std_bin = [[], [], [], [], []]
    confidence_levels = [0, 0.25, 0.5, 0.75, 1]
    for k in range(5):
        for connectivity_index in tqdm(range(10, 51), 
                         desc="Generating standard deviation plot for confidence level = " + str(confidence_levels[k])):
            bin = []
            for i in range(10):
                data = generate_data(size, number_of_positive_evidence, connectivity_index, confidence_levels[k])
                BPN = Network(*data)
                for j in range(30):
                    BPN.step()
                    believer_certainty = [agent.belief for agent in BPN.schedule.agents]
                bin.append(np.std(believer_certainty))
            std_bin[k].append(np.mean(bin))
    std_bin = np.array(std_bin)
    np.savetxt("std_versus_connectivity_data.txt", std_bin, delimiter=" ", fmt="%.7f")


def plot_std_versus_connectivity():
    '''This function generates the plot of standard deviation of belief distribution versus connectivity for different confidence levels'''
    plt.figure(figsize=(10, 6))
    std_bin = np.loadtxt("std_versus_connectivity_data.txt", delimiter=" ")
    confidence_levels = [0, 0.25, 0.5, 0.75, 1]
    for k in range(5):
        c = confidence_levels[k]
        plt.plot(np.arange(10, 51, 1), std_bin[k], label= f"global confidence level = {c}")
        # Add title and labels
        plt.title(f'belief distribution at 400 population size')
        plt.xlabel('connectivity index')
        plt.ylabel('standard deviation of belief distribution')
        plt.ylim(0, 0.015)
    plt.legend()
    plt.savefig("std_versus_k.png")


def std_versus_polarization_index_with_community_data(size, number_of_positive_evidence, connectivity_index):
    '''This function generates the data of standard deviation of belief distribution versus population size for different polarization index'''
    std_bin = [[], [], [], [], []]
    confidence_levels = [0, 0.25, 0.5, 0.75, 1]
    for k in range(5):
        for p in tqdm(np.arange(0.0, 1.0, 0.01), 
                         desc = "Generating standard deviation plot for confidence level = " + str(confidence_levels[k])):
            bin = []
            for i in range(10):
                data = generate_polarized_data_with_community(size, number_of_positive_evidence, connectivity_index, confidence_levels[k], p)
                BPN = Network(*data)
                for j in range(500):
                    BPN.step()
                    believer_certainty = [agent.belief for agent in BPN.schedule.agents]
                bin.append(np.std(believer_certainty))
            std_bin[k].append(np.mean(bin))
    std_bin = np.array(std_bin)
    np.savetxt("std_versus_polarization_index_with_community_data.txt", std_bin, delimiter=" ", fmt="%.7f")


def plot_std_versus_polarization_index_with_community():
    '''This function generates the plot of standard deviation of belief distribution versus polarization index for different confidence levels'''
    plt.figure(figsize=(10, 6))
    confidence_levels = [0, 0.25, 0.5, 0.75, 1]
    std_bin = np.loadtxt("std_versus_polarization_index_with_community_data.txt", delimiter=" ")
    for k in range(5):
        c = confidence_levels[k]
        plt.plot(np.arange(0.0, 1.0, 0.01), std_bin[k], label= f"global confidence level = {c}")
        # Add title and labels
        plt.title(f'belief distribution at connectivity k = 10')
        plt.xlabel('polarization index')
        plt.ylabel('standard deviation of belief distribution')
        # plt.ylim(0, 0.32)
    plt.legend()
    plt.savefig("std_versus_polarization_index_with_community.png")


def std_versus_polarization_index_data(size, number_of_positive_evidence, connectivity_index):
    '''This function generates the data of standard deviation of belief distribution versus population size for different polarization index'''
    std_bin = [[], [], [], [], []]
    confidence_levels = [0, 0.25, 0.5, 0.75, 1]
    for k in range(5):
        for p in tqdm(np.arange(0.0, 1.0, 0.01), 
                         desc = "Generating standard deviation plot for confidence level = " + str(confidence_levels[k])):
            bin = []
            for i in range(10):
                data = generate_polarized_data(size, number_of_positive_evidence, connectivity_index, confidence_levels[k], p)
                BPN = Network(*data)
                for j in range(50):
                    BPN.step()
                    believer_certainty = [agent.belief for agent in BPN.schedule.agents]
                bin.append(np.std(believer_certainty))
            std_bin[k].append(np.mean(bin))
    std_bin = np.array(std_bin)
    np.savetxt("std_versus_polarization_index_data.txt", std_bin, delimiter=" ", fmt="%.7f")


def plot_std_versus_polarization_index():
    '''This function generates the plot of standard deviation of belief distribution versus polarization index for different confidence levels'''
    plt.figure(figsize=(10, 6))
    confidence_levels = [0, 0.25, 0.5, 0.75, 1]
    std_bin = np.loadtxt("std_versus_polarization_index_data.txt", delimiter=" ")
    for k in range(5):
        c = confidence_levels[k]
        plt.plot(np.arange(0.0, 1.0, 0.01), std_bin[k], label= f"global confidence level = {c}")
        # Add title and labels
        plt.title(f'belief distribution at connectivity k = 10')
        plt.xlabel('polarization index')
        plt.ylabel('standard deviation of belief distribution')
    plt.legend()
    plt.savefig("std_versus_polarization_index.png")