# header file

import mesa ## This module allows us to create agent based models
import numpy as np   ## This module allows us to do array manipulations and math
import pandas as pd  ## This module allows us to import files
import networkx as nx ## This module allows us to create networks
import seaborn as sns ## This module allows us to create plots
import matplotlib.pyplot as plt ## This module allows us to create plots
import os ## This module allows us to interact with the operating system

class Believer(mesa.Agent):
    """A believer in this world with a unique structure of understanding."""

    # initialize the following
    # unique_id: an integer that represents the id of the agent
    # agent_type: a string that represents the type of the agent
    # model: a weighted social network that contains all the believers, a poll of evidence and critical events
    # belief: a float in [0,1] that represents the confidence of an agent on a particular statement
    # self_confidence: a float in [0,1] that represents the weight of the person on oneself's judgement
    # structure of understanding: an array of weights in [0,1] on the poll of evidence
    # confidence_level_on_evidence: an array of confidence level in [0,1] on the poll of evidence

    def __init__(self, unique_id, model, belief, agent_type, self_confidence,
                 structure_of_understanding, confidence_level_on_evidence):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.model = model
        self.belief = belief
        self.agent_type = agent_type
        self.structure_of_understanding = structure_of_understanding
        self.confidence_level_on_evidence = confidence_level_on_evidence
        self.self_confidence = self_confidence

    def get_belief(self):
        '''return the belief of the agent'''
        return self.belief
    
    def get_agent_type(self):
        '''return the type of the agent'''
        return self.agent_type
    
    def get_structure_of_understanding(self):
        '''return the structure of understanding of the agent'''
        return self.structure_of_understanding

    def get_confidence_level_on_evidence(self):
        '''return the confidence level on the poll of evidence of the agent'''
        return self.confidence_level_on_evidence
    
    def get_self_confidence(self):
        '''return the self confidence of the agent'''
        return self.self_confidence
    
    def get_unique_id(self):
        '''return the unique id of the agent'''
        return self.unique_id
    
    def update_evidence(self):
        '''update the level of certainty on the list of evidence based on self confidence level
           and its neighbors' confidence level, at the same time take into account the weights'''
        n = len(self.structure_of_understanding) # a length integer that is helpful for us to loop through
        # Get the list of neighbors' node IDs
        neighbors = self.model.grid.get_neighbors(self.unique_id)
        for neighbor in neighbors:
            neighbor_id = neighbor.unique_id
            # Get the weight of the edge between self and the neighbor
            weight = self.model.G[self.unique_id][neighbor_id]['weight']

            # Update the confidence level on each piece of evidence
            for i in range(n):
                self.confidence_level_on_evidence[i] = (
                    self.confidence_level_on_evidence[i] * (1 - weight) +
                    neighbor.confidence_level_on_evidence[i] * weight
                )
            # Update the confidence level on each pairs of opposite evidence
            for pair in self.model.opposite_evidence_pairs:
                i, j = pair
                self.confidence_level_on_evidence[i] = (
                    self.confidence_level_on_evidence[i] * (1 - weight) +
                    (1 - neighbor.confidence_level_on_evidence[j]) * weight
                )
                self.confidence_level_on_evidence[j] = (
                    self.confidence_level_on_evidence[j] * (1 - weight) +
                    (1 - neighbor.confidence_level_on_evidence[i]) * weight
                )

    def update_belief(self):
        '''update belief based on self reasoning and social norm'''
        n = len(self.structure_of_understanding) # a length integer that is helpful for us to loop through
        # initialize the portion of the belief that comes from self_reasoning
        self_reasoning = 0
        # initialize the portion of the belief that comes from social norm
        social_norm = 0
        # update self reasoning
        for i in range(n):
            self_reasoning += self.structure_of_understanding[i] * self.confidence_level_on_evidence[i]
        # update social norm
        neighbors = self.model.grid.get_neighbors(self.unique_id)
        for neighbor in neighbors:
            neighbor_id = neighbor.unique_id
            # Get the weight of the edge between self and the neighbor
            weight = self.model.G[self.unique_id][neighbor_id]['weight']
            social_norm += weight * neighbor.belief
            
        # update the current belief based on self confidence, social norm and self reasoning
        self.belief = self_reasoning * self.self_confidence + social_norm * (1 - self.self_confidence)

    def experience_critical_event(self):
        '''update evidence based on a critical event'''
        n = len(self.structure_of_understanding) # a length integer that is helpful for us to loop through
        for i in range(n):
            self.confidence_level_on_evidence[i] = min(1, self.confidence_level_on_evidence[i] * self.model.critical_event[3][i])

    # update evidence then belief at each time step
    def step(self):
        # update evidence
        self.update_evidence()
        # if critical event happens, update belief
        if (self.model.critical_events[1] == self.model.schedule.steps and self.unique_id in self.model.critical_events[2]):
          self.experience_critical_event()
        self.update_belief()

class Network(mesa.Model):
    """A network with some number of agents."""
    # initialize the following
    # A social network containing all agents
    # An array that stores the poll of evidence, with data type strings
    # An array that stores arrays of evidences and targeting population for critical event
    def __init__(self, critical_events, evidence_poll, opposite_evidence_pairs, social_network_filename,
                 believer_data_filename,
                 structure_of_understanding_data_filename,
                 confidence_level_on_evidence_data_filename):
        super().__init__()
        self.critical_events = critical_events
        self.evidence_poll = evidence_poll
        self.opposite_evidence_pairs = opposite_evidence_pairs
        # check if the files needed actually exist in the working directory
        if not os.path.exists(social_network_filename):
            print(f"File does not exist in the working directory: {social_network_filename}")
        if not os.path.exists(believer_data_filename):
            print(f"File does not exist in the working directory: {believer_data_filename}")
        if not os.path.exists(structure_of_understanding_data_filename):
            print(f"File does not exist in the working directory: {structure_of_understanding_data_filename}")
        if not os.path.exists(confidence_level_on_evidence_data_filename):
            print(f"File does not exist in the working directory: {confidence_level_on_evidence_data_filename}")
            
        # load the social network
        try:
            self.G = nx.read_weighted_edgelist(social_network_filename, nodetype=int, create_using=nx.Graph())
            print("Graph loaded successfully.")
            print("Graph", self.G)
        except Exception as e:
            print(f"Error loading graph: {e}")
        self.grid = mesa.space.NetworkGrid(self.G) # Initialize NetworkGrid with the graph
        self.critical_events = critical_events # a sequence of critical events
        self.evidence_poll = evidence_poll # a poll of evidence for people's belief
        self.schedule = mesa.time.RandomActivation(self) # create schedule
        # read in the agent type, belief and self confidence for each agent from a single file
        believer_data = pd.read_csv(believer_data_filename, sep=',', header = None).values
        belief = believer_data[:, 0]
        self_confidence = believer_data[:, 1]
        agent_type = believer_data[:, 2]
        # read in the structure of understanding and confidence level from separate files
        # with each row a structure of understanding or confidence level of an agent
        structure_of_understanding_data = pd.read_csv(structure_of_understanding_data_filename,
                                                      sep=',', header = None).values
        confidence_level_on_evidence_data = pd.read_csv(confidence_level_on_evidence_data_filename,
                                                        sep=',', header = None).values
        # Create and place agents on nodes
        for i in range(len(belief)):
            agent = Believer(i+1, self, belief[i], agent_type[i],
                             self_confidence[i], structure_of_understanding_data[i],
                             confidence_level_on_evidence_data[i])
            # add the agent to schedule
            self.schedule.add(agent)
            # place the agent onto the grid
            self.grid.place_agent(agent, i+1)
    
    def get_critical_events(self):
        '''return the critical events'''
        return self.critical_events
    
    def get_evidence_poll(self):
        '''return the poll of evidence'''
        return self.evidence_poll
    
    def get_opposite_evidence_pairs(self):
        '''return the opposite evidence pairs'''
        return self.opposite_evidence_pairs
    
    def get_agent(self, unique_id):
        '''return the agent with the unique_id'''
        return self.schedule.agents[unique_id]
    
    def generate_network_visualization(self, step, pos, believer_certainty):
        """Generate and save the network visualization for the given time step where 
        the size of node is proportional to the certainty on the belief."""
        # Create a new figure
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size as needed

        # Draw the graph with positions (e.g., spring layout)
        node_sizes = 1000000 * np.array(believer_certainty)**10  # Adjust the size of nodes
        nx.draw(self.G, pos, ax=ax, with_labels=True, node_color="skyblue", 
                edge_color="gray", node_size = node_sizes, font_size=10, font_color="black")

        # Set the title to indicate the current time step
        ax.set_title(f"Network at Time Step {step}")

        # Save the figure
        plt.savefig(os.path.join("output", f"network_step_{step}.png"))  # Saves the figure as a PNG file
        
        # Close the plot to free memory
        plt.close(fig)


    def generate_belief_distribution(self, step, believer_certainty):
        """Generate and save the plot of belief certainty distribution for the given time step."""
        # Create a new figure
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size as needed
        
        # Create a histogram with seaborn
        g = sns.histplot(believer_certainty, discrete=False)
        
        g.set(
            title="Belief distribution", xlabel="Certainty on Belief", ylabel="Number of agents"
        );  # The semicolon is just to avoid printing the object representation
        
        # Set the title to indicate the current time step
        ax.set_title(f"Belief distribution at Time Step {step}")

        # Save the figure
        plt.savefig(os.path.join("output", f"distribution_step_{step}.png"))  # Saves the figure as a PNG file
        
        # Close the plot to free memory
        plt.close(fig)

    # update the belief of each agent at each time step
    def step(self):
        self.schedule.step()