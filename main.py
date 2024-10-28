# main.py

from model import Network, Believer
import seaborn as sns ## This module allows us to create plots
import matplotlib.pyplot as plt ## This module allows us to create plots
import networkx as nx
import os
import cv2

# Set the path to the folder where your PNG images are stored
image_folder = 'path_to_your_images'
video_name = 'output_video.mp4'

# Get a sorted list of all PNG files in the folder
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()  # Sort the images to ensure they are in the correct order

# Read the first image to get frame dimensions
first_image_path = os.path.join(image_folder, images[0])
frame = cv2.imread(first_image_path)
height, width, layers = frame.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' or 'avc1' for different formats
fps = 30  # Frames per second
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# Loop through each image and add it to the video
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

# Release the video writer object
video.release()

print(f"Video saved as {video_name}")


def main():
    # read in the conditions of the network
    critical_events = [["chocolate commercial", 10, 1.5, 0.5, 1.3, 0.7, 0.8], [1, 3, 5, 7, 9]]
    evidence_poll = ("The level of sweetness of chocolate", "The level of bitterness of chocolate",
                    "The level of chewiness of chocolate", "The level of hardness of chocolate",
                    "The level of healthy effect of chocolate")
    opposite_evidence_pairs = ((0, 1), (2, 3))
    
    social_network_filename = "data/social_network.tsv"
    believer_data_filename = "data/believer_data.csv"
    structure_of_understanding_data_filename = "data/structure_of_understanding.csv"
    confidence_level_on_evidence_data_filename = "data/confidence_level_on_evidence.csv"

    # create the network
    BPN = Network(critical_events, evidence_poll, opposite_evidence_pairs, 
                  social_network_filename, believer_data_filename,
                  structure_of_understanding_data_filename,
                  confidence_level_on_evidence_data_filename)
    
    pos = nx.spring_layout(BPN.G)  # You can choose other layouts if needed
    # run it for 10 steps
    for i in range(100):
        BPN.step()
        if i%5 == 0:
            believer_certainty = [agent.belief for agent in BPN.schedule.agents]
            BPN.generate_belief_distribution(i, believer_certainty)
            BPN.generate_network_visualization(i, pos, believer_certainty)
            print(f"Step {i} completed.")

# Ensure main() is executed only when this script is run directly
if __name__ == "__main__":
    main()