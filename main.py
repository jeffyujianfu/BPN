# main.py

import simulation
import sys

def main(number_of_agents, number_of_positive_evidence, connectivity_index, global_self_confidence_level):
    # generate the data, if not, comment this line
    #data = simulation.generate_data(number_of_agents, number_of_positive_evidence, connectivity_index, global_self_confidence_level)
    #data1 = simulation.generate_polarized_data(number_of_agents, number_of_positive_evidence, connectivity_index, global_self_confidence_level, 0.2)
    #data2 = simulation.generate_polarized_data_with_community(number_of_agents, number_of_positive_evidence, connectivity_index, global_self_confidence_level, 0.2)
    
    # read in the conditions of the network
    critical_events = [["chocolate commercial", 10, 1.5, 0.5, 1.3, 0.7, 0.8], [1, 3, 5, 7, 9]]
    
    # generate the belief distribution plot
    #simulation.generate_belief_distribution(data)
    #simulation.generate_social_pressure_distribution(data)
    #simulation.generate_belief_distribution(data1, True, output_folder = "output_polarized_without_community")
    #simulation.generate_social_pressure_distribution(data1, output_folder = "output_polarized_without_community")
    
    # generate the belief distribution plot
    #simulation.generate_belief_distribution(data2, True, output_folder = "output_polarized_with_community")
    #simulation.generate_social_pressure_distribution(data2, output_folder = "output_polarized_with_community")
    
    #simulation.std_versus_population_size_data(number_of_positive_evidence, connectivity_index)
    
    #simulation.std_versus_connectivity_data(number_of_positive_evidence, number_of_agents)
    
    #simulation.std_versus_polarization_index_with_community_data(number_of_agents, number_of_positive_evidence, connectivity_index)
    
    simulation.plot_std_versus_polarization_index_with_community()

    #simulation.std_versus_polarization_index_data(number_of_agents, number_of_positive_evidence, connectivity_index)
    
    #simulation.plot_std_versus_polarization_index()
    
    #simulation.plot_std_versus_connectivity()
    
    #simulation.plot_std_versus_population_size()


# system modification for the main function to be called from the command line
if __name__ == '__main__':
    # Check if the correct number of arguments was provided
    if len(sys.argv) != 5:
        print("Usage: python3 main.py <number_of_agents> <number_of_positive_evidence> <connectivity_index> <global_self_confidence_level>")
        sys.exit(1)
    
    # Parse command-line arguments
    try:
        number_of_agents = int(sys.argv[1])
        number_of_positive_evidence = int(sys.argv[2])
        connectivity_index = int(sys.argv[3])
        global_self_confidence_level = float(sys.argv[4])
        0 <= global_self_confidence_level <= 1
    except ValueError:
        print("Please provide integer values for all 3 arguments and a float between 0 to 1 for the last one.")
        sys.exit(1)

    # Call the main function with parsed arguments
    main(number_of_agents, number_of_positive_evidence, connectivity_index, global_self_confidence_level)