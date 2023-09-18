from shell.fleet.network import TopologyGenerator
import os

# Directory to save generated graph files
OUTPUT_DIR = "generated_graphs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define some parameters to iterate over
node_counts = [4, 8, 16, 32]
# node_counts = [4]
edge_probs = [0.0, 0.2, 0.4, 0.6, 1.0]

# Create an instance of your topology generator for each parameter set
for num_nodes in node_counts:
    for edge_prob in edge_probs:
        generator = TopologyGenerator(num_nodes, edge_prob)
        
        # Generate various topologies and save them
        # G_fully_connected = generator.generate_fully_connected()
        # G_ring = generator.generate_ring()
        G_random = generator.generate_random()

        

        # TopologyGenerator.plot_graph(G_fully_connected, layout="spring",
        #                              save_path=os.path.join(OUTPUT_DIR, f"fully_connected_nodes_{num_nodes}_edgeprob_{edge_prob}.png"))
        # TopologyGenerator.plot_graph(G_ring, layout="circular",
        #                               save_path=os.path.join(OUTPUT_DIR, f"ring_nodes_{num_nodes}_edgeprob_{edge_prob}.png"))
        TopologyGenerator.plot_graph(G_random, layout="random",
                                      save_path=os.path.join(OUTPUT_DIR, f"rand_nodes_{num_nodes}_edgeprob_{edge_prob}.png"))
