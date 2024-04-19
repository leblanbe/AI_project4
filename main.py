"""
Group Number 2
Marcus Kamen, Marius Schueller, Brynn LeBlanc, and Daniel Yakubu

Use your own API key for ChatGPT, as described under the imports

To run this file you press play on main if using Pycharm. Otherwise, just run the main function with any
standard python interpreter. Make sure to have math, random, csv, queue, PriorityQueue, copy, time, and openai
installed as packages on your computer for python. During runtime, you will be asked for the start city which should be
formatted in the form CityStateInitials (for example NashvilleTN), in addition to the file paths where the road trip
data will be obtained, the average speed you want to drive, and the file path to save the roadtrip.

We decided to use an A* type search for our solution. The utility function combines both the preference
values of the nodes and edges as well as the distance of the traveled to node from the start location, or the required locations if those are used.
We used both of these values so that we could create a search that moves further away from the start for the
first half of the road trip, and moves closer to the start for the second half, while incorporating preference
values to help. In addition, the search moves towards the requried locations quickly so they are always accessed.
We thought that this was a good strategy so that the road trip would go out from the start via
a high preferences, and then return to the start in a circular manner.
During our A* search we were able to use both path cost and heuristic distance by weighting the preference
value of each node and edge. We did this because the different edge values were between 21 and 338,
thereby putting the influence of preferences on being slightly less than the average distance, but still significant.
We run our road trip through the ChatGPT LLM API. The prompt engineering for our results is described in our report and our Give_Narrative function.
"""

import math
import random
import csv
from queue import PriorityQueue
import copy
import time
import openai

# this imports a private API key
import private_info

# either replace "private_info.api_key" with your API key, or update the private_info file to include your API key
client = openai.OpenAI(api_key=private_info.api_key)

class Node:
    """
        Class representation of a Node (a location in the road network)

        self.name       -- name of location
        self.x          -- latitude of location
        self.y          -- longitude of location
        self.preference -- preference values of location
        self.themes     -- themes for each location
    """

    def __init__(self, name, x, y):
        """
            Initialize instance of Node

            :param name: self.name copy
            :param x:    self.x copy
            :param y:    self.y copy
        """
        self.name = name
        self.x = x
        self.y = y
        self.preference = 0
        self.themes = set() #Prevent duplicate themes

    def time_at_location(self):
        """
            Get the time at this location based on the preference value. Increasing function
            that sets the 0 value at 5

            :return: time at this location
        """
        return self.preference * 100 + 5

    def __hash__(self):
        return hash(self.name)
    
    def assign_themes(self, themes):
        """
            Assign a random subset of themes to the node.
            
            :param themes: valid themes to choose between
        """
        num_themes = random.randint(0, 10)  # Random number of themes from 0 to 10
        self.themes = set(random.sample(themes, num_themes))


class Edge:
    """
        Class representation of an edge between locations

        self.label          -- name of edge
        self.locationA      -- first location of edge
        self.locationB      -- second location of edge
        self.actualDistance -- distance between locations of edge
        self.preference     -- preference of edge
        self.themes         -- themese for each edge
    """

    def __init__(self, label, locationA, locationB, actualDistance):
        """
            Initialize instance of edge

            :param label:          self.label copy
            :param locationA:      self.locationA copy
            :param locationB:      self.locationB copy
            :param actualDistance: self.actualDistance copy
        """
        self.label = label
        self.locationA = locationA
        self.locationB = locationB
        self.actualDistance = actualDistance
        self.preference = 0
        self.themes = set()

    def add_time_on_edge(self, x):
        """
            Adds time for traveling edge (currently just calls time_at_location)

            :param x: speed to travel edge
            :return:  time at location for edge
        """
        return self.time_at_location(x)

    def time_at_location(self, x):
        """
            Gets the edge time at location
            :param x: speed to traverse edge
            :return: distance divided by speed
        """
        return self.actualDistance / x

    def assign_themes(self, themes):
        """
            Assign a random subset of themes to the node.
            
            :param themes: valid themes to choose between
        """
        num_themes = random.randint(0, 10)  # Random number of themes from 0 to 10
        self.themes = set(random.sample(themes, num_themes))


class Roadtrip:
    """
        Class representation of a road trip (can be partial trip)

        self.NodeList           -- List of locations in trip
        self.EdgeList           -- List of edges in trip
        self.currentTimeElapsed -- Total time elapsed in trip
        self.time_search        -- Time in took to search this trip
        self.startNode          -- Start node of road trip
    """

    def __init__(self):
        """
            Initialize an instance of a Roadtrip, sets all fields to 0
        """
        self.NodeList = []
        self.EdgeList = []
        self.currentTimeElapsed = 0
        self.time_search = 0
        self.startNode = None

    def __lt__(self, other):
        """
            Overloaded less than operator for road trips
            Needed in case library functions use less than operators to break ties between equal utilities
            in PriorityQueue

            :param other: other Roadtrip working with
            :return: boolean if self < other
        """
        return self.total_preference() < other.total_preference()
    
    def total_theme_count(self):
        """
            Gets the total count of all themes in this trip
            
            :return: dictionary of theme counts for each theme index
        """
        themes = {}         
        visited = set()

        for node in self.NodeList:
            if node != self.startNode and node not in visited:
                visited.add(node)
                for theme in node.themes:
                    if theme in themes.keys():
                        themes[theme] += 1
                    else:
                        themes[theme] = 0
             
        for edge in self.EdgeList:
            if edge not in visited:
                visited.add(edge)
                for theme in edge.themes:
                    if theme in themes.keys():
                        themes[theme] += 1
                    else:
                        themes[theme] = 0
        
        return themes

    def total_preference(self):
        """
            Gets the total preference of all nodes and edges in Roadtrip

            :return: total preference
        """
        visited = set()
        preference = 0.0
        for node in self.NodeList:
            if node != self.startNode and node not in visited:
                preference = preference + node.preference
                visited.add(node)

        for edge in self.EdgeList:
            if edge not in visited:
                preference = preference + edge.preference
                visited.add(edge)

        return preference

    def get_total_distance(self):
        """
            Gets the total distance traveled on the road trip

            :return: sum of distances of all edges
        """
        return sum(edge.actualDistance for edge in self.EdgeList)

    def time_estimate(self, x):
        """
            Gets the time estimate for the full Roadtrip
            

            :param x: speed to travel edges
            :return: total time of Roadtrip
        """
        visited = set()
        time = 0.0
        for node in self.NodeList:
            if node != self.startNode and node not in visited:
                time = time + node.time_at_location()
                visited.add(node)

        for edge in self.EdgeList:
            if edge not in self.EdgeList:
                time = time + edge.add_time_on_edge(x)
                visited.add(edge)

        return time

    def hasNode(self, node):
        """
            Checks if a node is present in this Roadtrip

            :param node: node to check
            :return: if node is present in NodeList
        """
        for nod in self.NodeList:
            if nod.name == node.name:
                return True
        return False

    def hasNodeString(self, node):
        """
            Checks if a node is present in this Roadtrip

            :param node: node to check as a string name
            :return: if node is present in NodeList
        """
        for nod in self.NodeList:
            if nod.name == node:
                return True
        return False
    
    def hasEdge(self, edge):
        """
            Checks if an edge is present in this Roadtrip
            
            :param edge: edge to check
            :return: if edge is present in EdgeList
        """
        for ed in self.EdgeList:
            if ed.label == edge.label:
                return True
        return False

    def get_node_by_location(self, node_name):
        """
            Gets a node based on its name

            :param node_name: name of node
            :return: node with that name
        """
        for node in self.NodeList:
            if node.name == node_name:
                return node
        raise ValueError(f"No node {node_name} in road trip network")

    def find_NodeB(self, edge):
        """
            Finds the second node listed in an edge

            :param edge: Edge to check
            :return: Node corresponding to second node of edge
        """
        for node in self.NodeList:
            if edge.locationB == node.name:
                return node

    def find_NodeA(self, edge):
        """
            Finds the first node listed in an edge

            :param edge: Edge to check
            :return: Node corresponding to first node of edge
        """
        for node in self.NodeList:
            if edge.locationA == node.name:
                return node                                                                     
    
    def print_result(self, num, start_node, maxTime, speed_in_mph, themes):
        """
            Print the results of a road trip.

            :param start_node: (node or string) The starting node or location. If a string is provided,
                                it will be used to fetch the corresponding Node using get_node_by_location.
            :param maxTime: (float) The maximum time allowed for the trip.
            :param speed_in_mph: (float) The speed in miles per hour used for time estimation.
            :param themes: list of themes in order
            :return: None

            Prints the simulation results, including routing details and summary information,
            to the console. The output includes the starting node, maximum time, speed, routing
            details, and summary information such as total preference, total distance, and
            estimated time.

            Example:
            ```
            router = YourRouterClass()
            router.print_result("StartLocation", 10.0, 60.0)
            ```
        """
        if not isinstance(start_node, Node):
            start_node = self.get_node_by_location(start_node)

        cur_node = start_node
        line_number = 1

        print(f"Solution {num}", end=" ")
        print(start_node.name, end=" ")
        print(maxTime, end=" ")
        print(speed_in_mph, end=" ")
        print("\n")

        for edge in self.EdgeList:

            print(line_number, ".", end=" ")

            print(cur_node.name, end=" ")

            if self.find_NodeA(edge) == cur_node:
                cur_node = self.find_NodeB(edge)
            else:
                cur_node = self.find_NodeA(edge)

            print(cur_node.name, end=" ")
            print(edge.label, end=" ")
            print(edge.preference, end=" ")
            print(edge.add_time_on_edge(speed_in_mph), end=" ")
            print(cur_node.preference, end=" ")
            print(cur_node.time_at_location(), end=" ")
            print("\n")
            line_number += 1

        print(start_node.name, end=" ")
        print(self.total_preference(), end=" ")
        print(self.get_total_distance(), end=" ")
        print(self.time_estimate(speed_in_mph), end=" ")
        print()
        print(self.total_theme_count())
        print(themes)

    def write_result_to_file(self, num, start_node, maxTime, speed_in_mph, themes, output_file=None):
        """
            Write the results of a round trip to a file.

            :param start_node: (Node or string) The starting node or location. If a string is provided,
                                                it will be used to fetch the corresponding Node using
                                                get_node_by_location.
            :param maxTime: (float) The maximum time allowed for the trip.
            :param speed_in_mph: (float) The speed in miles per hour used for time estimation.
            :param output_file: (str, optional) The name of the file to write the results to.
                                If not provided, the default filename is "default_output.txt".
            :param themes: Themes listed in order
            :return: None

            Writes the simulation results, including routing details and summary information,
            to the specified output file. The file includes the starting node, maximum time,
            speed, routing details, and summary information such as total preference and
            estimated time.

            Example:
            ```
            router = YourRouterClass()
            router.write_result_to_file("StartLocation", 10.0, 60.0, "output.txt")
            ```
        """
        if not isinstance(start_node, Node):
            start_node = self.get_node_by_location(start_node)

        cur_node = start_node
        line_number = 1

        if output_file is None:
            output_file = "default_output.txt"

        with open(output_file, 'a', encoding='utf-8') as file:
            file.write(f"Solution{num} ")
            file.write(f"{start_node.name} ")
            file.write(f"{maxTime} ")
            file.write(f"{speed_in_mph} ")
            file.write("\n")

            for edge in self.EdgeList:
                file.write(f"{line_number}. ")
                file.write(f"{cur_node.name} ")

                if self.find_NodeA(edge) == cur_node:
                    cur_node = self.find_NodeB(edge)
                else:
                    cur_node = self.find_NodeA(edge)

                file.write(f"{cur_node.name} ")
                file.write(f"{edge.label} ")
                file.write(f"{edge.preference} ")
                file.write(f"{edge.add_time_on_edge(speed_in_mph)} ")
                file.write(f"{cur_node.preference} ")
                file.write(f"{cur_node.time_at_location()} ")
                file.write("\n")
                line_number += 1

            file.write(f"{start_node.name} ")
            file.write(f"{self.total_preference()} ")
            file.write(f"{self.get_total_distance()} ")
            file.write(f"{self.time_estimate(speed_in_mph)} ")
            file.write("\n")
            file.write(str(self.total_theme_count()))
            file.write("\n")
            file.write(str(themes))
            file.write("\n")
            file.write("\n")


    def Give_Narrative(self, start_node, output_file):
        """
            Gives the road trip narrative for this road trip
            
            :param start_node: the start node of the road trip for initialization of reporting
            :param output_file: the file to print the narrative to
        """

        if not isinstance(start_node, Node):
            start_node = self.get_node_by_location(start_node)

        cur_node = start_node
        line_number = 1
        
        """
            Format
            i = line_number
            name1 = first node
            name2 = second node
            
            i. name1 (name1.preference, name1.themes) -> (edge.actualDistance, edge.preference, edge.themes) name2 (name2.preference, name2.themes)\n
        """

        user_content = ""

        for edge in self.EdgeList:
            user_content += str(line_number) + ". "
            user_content += cur_node.name
            user_content += f" ({str(cur_node.preference)}, {str(cur_node.themes)})"
            user_content += f" -> ({str(edge.actualDistance)}, {str(edge.preference)}, {str(edge.themes)}) "

            if self.find_NodeA(edge) == cur_node:
                cur_node = self.find_NodeB(edge)
            else:
                cur_node = self.find_NodeA(edge)

            user_content += cur_node.name
            user_content += f" ({str(cur_node.preference)}, {str(cur_node.themes)})"

            user_content += "\n"
            line_number += 1

        # list of patterns used:
        # 
        # meta language creation pattern
        # When I say X, I mean Y (or would like you to do Y)
        # 
        # persona pattern
        # Act as persona X
        # Provide outputs that persona X would create  
        #
        # alternative approaches pattern
        # Within scope X, if there are alternative ways to accomplish the same thing, list the best alternate approaches
        # (Optional) compare/contrast the pros and cons of each approach
        # (Optional) include the original way that I asked
        # (Optional) prompt me for which approach I would like to use  
       
        system_content = ("You are a road trip assistant. You give attractions at various locations as well "
                         "as preferred edges based on a road trip. Provide the output that a road trip assistant would create. "
                         "When I say 'give me a road trip assistant report,' I mean give me a report based on my inputted road trip "
                         "with the role of a road trip assistant.")
        
        assistant_message = ("Road trips will be in the following pattern."
                            "i. name1 (name1.preference, name1.themes) -> (edge.actualDistance, edge.preference, edge.themes) name2 (name2.preference, name2.themes)\n"
                            "where i is a number, name1 is a location, name2 is another location, and -> refers to an edge between the two locations.\n"
                            "Note that higher preference means that the location should be more preferred in the road trip, so make sure to explain those locations better")
                 
        prompt = ("Give me a road trip assistant report. If there are alternative ways to accomplish this, give two alternative approaches "
                 "and compare and contrast these approaches.")

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "assistant",
                    "content": assistant_message
                },
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            temperature=1,
            max_tokens=3000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        
        output = response.choices[0].message.content

        print("\nLLM Generated Response: ")
        print(output)
        
        with open(output_file, 'a', encoding='utf-8') as file:
            file.write("\nLLM Generated Response:\n")
            file.write(output)
            file.write("\n")
            file.write("\n")
            

class RegressionTree:
    """
        Representation of a regression tree for utilities
        
        self.root() -- root of tree (tree is recursively defined)
    """
    
    def __init__(self, max_depth=4):
        """
            Initialize regression tree
        """
        self.root = None

    def fit1(self):

        """
        Constructs a regression tree with predetermined node structure and values.

        The tree is constructed with hardcoded splits and prediction values.

        Note that even though the predicted leaf values are hard coded, they are not a single number.
        Instead, they predict random numbers over certain thresholds centered around the leaf values in
        the diagram below.

        The tree structure is as follows:
                                [0]
                      /                      \\
                    [1]                     [2]
                /        \             /           \\
            [3]         [4]          [5]           [6]
            /  \        / \          / \         /     \\
          [7]  [0.21] [0.3][0.62] [0.5] [0.61] [0.73] [0.92]
         /  \\
      [0.1] [0.5]

        Returns:
            None
        """


        self.root = RegressionNode(feature=0)
        self.root.left = RegressionNode(feature=1, value = None)
        self.root.right = RegressionNode(feature=2, value = None)

        self.root.left.left = RegressionNode(feature=3, value = None)
        self.root.left.right = RegressionNode(feature=4, value = None)

        self.root.right.left = RegressionNode(feature=5, value = None)
        self.root.right.right= RegressionNode(feature=6, value = None)

        self.root.left.left.left = RegressionNode(feature=7, value = None)
        self.root.left.left.right = RegressionNode(value = random.uniform(0.18, 0.24))

        self.root.left.right.left = RegressionNode(value = random.uniform(0.28, 0.32))
        self.root.left.right.right = RegressionNode(value = random.uniform(0.58, 0.66))

        self.root.right.left.left = RegressionNode(value = random.uniform(0.48, 0.52))
        self.root.right.left.right = RegressionNode(value = random.uniform(0.56, 0.66))

        self.root.right.right.left= RegressionNode(value = random.uniform(0.70, 0.76))
        self.root.right.right.right= RegressionNode(value = random.uniform(0.84, 1))


        self.root.left.left.left.left = RegressionNode(value = random.uniform(0, 0.2))
        self.root.left.left.left.right = RegressionNode(value = random.uniform(0.4, 0.6))


    def fit2(self):
        """
        Constructs a regression tree with predetermined node structure and values.

        The tree is constructed with hardcoded splits and prediction values.

        Note that even though the predicted leaf values are hard coded, they are not a single number.
        Instead, they predict random numbers over certain thresholds centered around the leaf values in
        the diagram below.

        The tree structure is as follows:
                                [0]
                      /                      \\
                     [1]                     [1]
                /           \             /           \\
            [2]            [2]          [2]           [2]
            /  \           / \          / \         /     \\
          [0.5]  [0.21] [0.3][0.62] [0.5] [0.61] [0.73] [0.92]

        Returns:
            None
        """


        self.root = RegressionNode(feature=0)
        self.root.left = RegressionNode(feature=1, value = None)
        self.root.right = RegressionNode(feature=1, value = None)

        self.root.left.left = RegressionNode(feature=2, value = None)
        self.root.left.right = RegressionNode(feature=2, value = None)

        self.root.right.left = RegressionNode(feature=2, value = None)
        self.root.right.right= RegressionNode(feature=2, value = None)

        self.root.left.left.right = RegressionNode(value = random.uniform(0.18, 0.24))
        self.root.left.left.left = RegressionNode(value = random.uniform(0.4, 0.6))

        self.root.left.right.left = RegressionNode(value = random.uniform(0.28, 0.32))
        self.root.left.right.right = RegressionNode(value = random.uniform(0.58, 0.66))

        self.root.right.left.left = RegressionNode(value = random.uniform(0.48, 0.52))
        self.root.right.left.right = RegressionNode(value = random.uniform(0.56, 0.66))

        self.root.right.right.left= RegressionNode(value = random.uniform(0.70, 0.76))
        self.root.right.right.right= RegressionNode(value = random.uniform(0.84, 1))
        

    def predict(self, sample):

        """
        Predicts the output value for a given sample using the fitted regression tree.

        If the tree has not been previously fitted, it will fit the tree before making predictions.
        Note: It makes no difference if `fit` is called multiple times.

        Args:
            sample (list): The input sample for which the prediction is made.

        Returns:
            float: The predicted output value for the given sample.
        """
        return self.traverse_tree(sample, self.root)

    def traverse_tree(self, sample, node):

        """
        Traverses the fitted regression tree recursively to predict the output value for a given sample.

        If the current node is a leaf node, the prediction value of the node is returned.
        Otherwise, the method recursively traverses the left or right subtree based on the feature value of the sample.

        Args:
            sample (list): The input sample for which the prediction is made.
            node (RegressionNode): The current node in the regression tree.

        Returns:
            float: The predicted output value for the given sample.
        """

        if node.is_leaf_node():
            return node.value
        if sample[node.feature] == 0:
            return self.traverse_tree(sample, node.left)
        else:
            return self.traverse_tree(sample, node.right)
        


class RegressionNode:
    """
    A class representing a node in a decision tree.

    Attributes:
    - feature (int or None): The index of the feature used for splitting at this node.
    - threshold (float or None): The threshold value used for splitting the feature.
    - left (Node or None): The left child node.
    - right (Node or None): The right child node.
    - value (int or None): The class label assigned to this node if it is a leaf node.

    Methods:
    - is_leaf_node(): Returns True if the node is a leaf node, False otherwise.

    The Node class represents a node in a decision tree. Each node contains information about
    the splitting feature, threshold, child nodes, and assigned class label (if it's a leaf node).

    Constructor Parameters:
    - feature (int or None): Index of the feature used for splitting at this node.
    - threshold (float or None): Threshold value used for splitting the feature.
    - left (Node or None): Left child node.
    - right (Node or None): Right child node.
    - value (int or None, optional): Class label assigned to this node if it's a leaf node.

    If the value parameter is provided, it indicates that the node is a leaf node, and the
    decision tree stops splitting further.

    Example:
    >>> node = Node(feature=0, threshold=2.5, left=Node(value=1), right=Node(value=0))
    >>> node.is_leaf_node()
    False
    """

    def __init__(self, feature=None, left=None, right=None, *, value=None):
        """
            Initialize Regression Tree node
            
            :param feature: self.feature copy
            :param threshold: self.threshold copy
            :param left: self.left copy
            :param value: self.value copy
        """
        self.feature = feature
        self.left = left
        self.right = right
        self.value = value
        

    def is_leaf_node(self):
        """
        Check if the node is a leaf node.

        Returns:
        - bool: True if the node is a leaf node, False otherwise.
        """
        return self.value is not None


class Roadtripnetwork:
    """
            Representation of a Roadtripnetwork object with information about the road network,
            including start location, file paths, maximum time allowed, speed, and result file.

            self.NodeList   -- List of nodes in network
            self.EdgeList   -- List of edges in network
            self.startLoc   -- Start location of search
            self.LocFile    -- File to find locations
            self.EdgeFile   -- File to find edges
            self.ThemeFile  -- File to find themes
            self.maxTime    -- Max time of road trip
            self.x_mph      -- Time to traverse an edge
            self.resultFile -- Where to output results
            self.startNode  -- Node corresponding to self.startLoc
            self.solutions  -- Road trip solutions currently found
            self.max_trials -- Maximum number of road trips the user wants to find
            self.forbidden_locations        -- Locations forbidden by the user
            self.required_locations         -- Locations required by the user
            self.index_in_required_checked  -- Variable to determine which required locations have been visited so far
            self.next_required_node         -- Variable to determine which required location will be attempted to be visited next
            self.Regression_tree            -- Regression tree to use to determine utilities
            self.themes                     -- Themes extracted from the theme file
    """

    def __init__(self, startLoc, LocFile, EdgeFile, ThemeFile, maxTime, x_mph, resultFile, max_trials, forbidden_locations, required_locations):
        """
            Initialize a Roadtripnetwork object

            :param startLoc:    self.startLoc copy
            :param LocFile:     self.LocFile copy
            :param EdgeFile:    self.EdgeFile copy
            :param ThemeFile:   self.ThemeFile copy
            :param maxTime:     self.maxTime copy
            :param x_mph:       self.x_mph copy
            :param resultFile:  self.resultList copy
            :param max_trails:  self.max_trails copy
            :param forbidden:   self.forbidden_locations copy
            :param required:    self.required_locations copy
        """
        self.NodeList = []
        self.EdgeList = []
        self.startLoc = startLoc
        self.LocFile = LocFile
        self.EdgeFile = EdgeFile
        self.ThemeFile = ThemeFile
        self.maxTime = maxTime
        self.x_mph = x_mph
        self.resultFile = resultFile
        self.startNode = None
        self.solutions = PriorityQueue()
        self.max_trials = max_trials
        self.forbidden_locations = forbidden_locations
        self.required_locations = required_locations
        self.index_in_required_checked = 0
        self.next_required_node = None
        self.regression_tree = None
        self.themes = []

    def location_preference_assignments(self, a=0.0, b=1.0):
        """
                Assign random preferences to all nodes in the road network within a specified range.

                :param a: Lower bound of the preference range.
                :param b: Upper bound of the preference range.
        """
        for node in self.NodeList:
            node.preference = a + self.regression_tree.predict(self.theme_indicator_vector(node)) * (b - a)

    def edge_preference_assignments(self, a=0.0, b=0.1):
        """
                Assign random preferences to all edges in the road network within a specified range.

                :param a: Lower bound of the preference range.
                :param b: Upper bound of the preference range.
        """
        for edge in self.EdgeList:
            edge.preference = a + self.regression_tree.predict(self.theme_indicator_vector(edge)) * (b - a)
    
    def assign_themes(self):
        
        """
            Assign themes to nodes and edges in the graph.
        """

        for node in self.NodeList:
            node.assign_themes(self.themes)
        
        for edge in self.EdgeList:
            edge.assign_themes(self.themes)

    def theme_indicator_vector(self, node_or_edge):

        """
        Generates a binary indicator vector representing the presence or absence of each theme.

        Args:
            node_or_edge: The node or edge for which the indicator vector is generated.

        Returns:
            list: A binary indicator vector where each element represents the presence (1) or absence (0)
                of a theme. The order of elements corresponds to the order of themes in the graph.
        """

    
        theme_presence_indicator = []
        
        # Check each theme
        for theme in self.themes:
            # Check if the theme exists at the node or edge
            if theme in node_or_edge.themes:
                theme_presence_indicator.append(1)  # Theme exists
            else:
                theme_presence_indicator.append(0)  # Theme does not exist
                
        return theme_presence_indicator


    def parseNodes(self):
        """
            Parse nodes from the CSV file and create Node objects for each location in the road network.
        """

        file_path = self.LocFile

        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                location_info = {
                    'Location Label': row['Location Label'],
                    'Latitude': float(row['Latitude']),
                    'Longitude': float(row['Longitude']),
                }

                # USE NODE CLASS
                self.NodeList.append(Node(location_info['Location Label'], location_info['Latitude'],
                                          location_info['Longitude']))

    def parseEdges(self):
        """
            Parse edges from the CSV file and create Edge objects for each connection in the road network.
        """

        file_path = self.EdgeFile  # Replace with your actual file path

        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                location_info = {
                    'edgeLabel': row['edgeLabel'],
                    'locationA': row['locationA'],
                    'locationB': row['locationB'],
                    'actualDistance': float(row['actualDistance']),
                }

                # USE EDGE CLASS
                self.EdgeList.append(Edge(location_info['edgeLabel'], location_info['locationA'],
                                          location_info['locationB'], location_info['actualDistance']))
             

    def parseThemes(self):
        """
            Parse themes from the CSV file and create an array of valid themes
        """
        
        file_path = self.ThemeFile
        
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                theme_info = {
                    'theme': row['']
                }
                self.themes.append(theme_info['theme'])
                

    def loadFromFile(self):
        """
            Loads data from files calling parseNodes(), parseEdges(), and parseThemes()
        """
        self.parseNodes()
        self.parseEdges()
        self.parseThemes()
        
    def initializeTree1(self):
        """
            Initializes and fits the regression tree to our first hand crafted tree
        """
        self.regression_tree = RegressionTree()
        self.regression_tree.fit1() # "Train" tree so it is ready to use
        
    def initializeTree2(self):
        """
            Initializes and fits the regression tree to our second hand crafted tree
        """
        self.regression_tree = RegressionTree()
        self.regression_tree.fit2() # "Train" tree so it is ready to use
        

    def initializeForSearch(self, tree):
        """
            Initializes the start node and assigns preferences before starting the search algorithm
            
            :param tree: The Regression tree to intialize (1 or 2)
        """
        if tree == 1:
            self.initializeTree1()
        else:
            self.initializeTree2()
        
        self.location_preference_assignments()
        self.edge_preference_assignments()
        
        for node in self.required_locations:
            if not node == '' and not node == ' ':
                present = False
                for nod in self.NodeList:
                    if node == nod.name:
                        present = True
                 
                if not present:
                    print(node)
                    raise ValueError('Required Locations Contains Invalid Node')
            
        for node in self.forbidden_locations:
            if not node == '' and not node == ' ':
                present = False
                for nod in self.NodeList:
                    if node == nod.name:
                        present = True
                 
                if not present:
                    print(node)
                    raise ValueError('Forbidden Locations Contains Invalid Node')
                

        present = False
        for node in self.NodeList:
            if self.startLoc == node.name:
                self.startNode = node
                present = True
                break
            
        if not present:
            raise ValueError('Invalid Start Node')
        
        if self.index_in_required_checked < len(self.required_locations) and not self.required_locations[self.index_in_required_checked] == ' ':
            for node in self.NodeList:
                if self.required_locations[0] == node.name:
                    self.next_required_node = node
                    break
                

    def astar_search(self):
        """
            Perform A* search to find an optimal path considering preferences and distances as evenly as possible.
            Updates the Roadtrip with the discovered path.
        """

        search_start = time.time()
        numSearches = 0
        frontier = PriorityQueue()
        trip = Roadtrip()
        trip.NodeList.append(self.startNode)
        trip.startNode = self.startNode
        frontier.put((0, trip))

        while numSearches < self.max_trials and (not frontier.empty()):
            trip = frontier.get()[1]
            # check if start node returned to
            if len(trip.EdgeList) > 1:
                if (self.find_NodeA(trip.EdgeList[len(trip.EdgeList) - 1]) == self.startNode
                        or self.find_NodeB(trip.EdgeList[-1]) == self.startNode):
                    trip.time_search = time.time() - search_start
                    self.solutions.put((-trip.total_preference(), trip))
                    numSearches = numSearches + 1
                    search_start = time.time()
                    continue

            # go through edge list and find related nodes
            for edge in self.EdgeList:
                name = trip.NodeList[len(trip.NodeList) - 1].name
                if edge.locationA == name:
                    node = self.find_NodeB(edge)
                    util = self.utility(trip, edge, node)
                    if not util == float('inf'):
                        newTrip = copy.deepcopy(trip)
                        newTrip.NodeList.append(node)
                        newTrip.EdgeList.append(edge)
                        frontier.put((util, newTrip))
                elif edge.locationB == name:
                    node = self.find_NodeA(edge)
                    util = self.utility(trip, edge, node)
                    if not util == float('inf'):
                        newTrip = copy.deepcopy(trip)
                        newTrip.NodeList.append(node)
                        newTrip.EdgeList.append(edge)
                        frontier.put((util, newTrip))

    def utility(self, trip, edge, node):
        """
            Calculate the utility value for a given edge and node based on distance, preferences, and time constraints.

            :param trip: Trip object
            :param edge: Edge object representing the road segment.
            :param node: Node object representing the location.
            :return: Utility value considering distance to the start, node and edge preferences.
        """

        timeEstimate = trip.time_estimate(self.x_mph)
        
        # give better preference to trips with required locations
        adjustment = 0
        for nod in self.required_locations:
            if trip.hasNodeString(nod):
                adjustment -= 100000
        
        if self.index_in_required_checked < len(self.required_locations) and not self.required_locations[self.index_in_required_checked] == '':
            distToLocation = math.sqrt(math.pow(node.x - self.next_required_node.x, 2) + math.pow(node.y - self.next_required_node.y, 2))
            
            if node.name in self.forbidden_locations or timeEstimate > self.maxTime:
                return float('inf')
            
            if node.name == self.required_locations[self.index_in_required_checked]:
                self.index_in_required_checked = self.index_in_required_checked + 1
                if self.index_in_required_checked < len(self.required_locations) and not self.required_locations[self.index_in_required_checked] == ' ':
                    for node in self.NodeList:
                        if self.required_locations[self.index_in_required_checked] == node.name:
                            self.next_required_node = node
                            break
                return -float('inf')
            
            if timeEstimate < self.maxTime:
                distToLocation = distToLocation - 1000
            else:
                return float('inf')

            if trip.hasNode(node):
                return 100 * distToLocation + 10 - 10 * self.trip_utility_with_new_node_edge(trip, node, edge) + adjustment
            return 100 * distToLocation - 10 * self.trip_utility_with_new_node_edge(trip, node, edge) + adjustment


        distToStart = math.sqrt(math.pow(node.x - self.startNode.x, 2) + math.pow(node.y - self.startNode.y, 2))
        # If the node is in the forbidden locations, return infinite utility to avoid selecting it
        if node.name in self.forbidden_locations or timeEstimate > self.maxTime:
            return float('inf')
        
        if node.name in self.required_locations and not trip.hasNode(node):
            return -float('inf')

        # Adjust distance to start based on whether the trip is in the first half or the second half of the max time
        if timeEstimate < self.maxTime / 2:
            distToStart = distToStart * (-1)
        else:
            distToStart = distToStart - 1000

        if trip.hasNode(node):
            return distToStart + 10 - 10 * self.trip_utility_with_new_node_edge(trip, node, edge) + adjustment
        return distToStart - 10 * self.trip_utility_with_new_node_edge(trip, node, edge) + adjustment
    
    def trip_utility_with_new_node_edge(self, trip, node, edge):
        """
            Calculate the utility of the trip based only on node and edge preferences
            This function also includes an extra node and edge to consider so that the utility
            function can consider those extra values to determine if it is to be added to the road trip
            New node and edge preferences are only considered if those nodes and edges are not already in the trip
            
            Note: Trip overall utility from each edge and location preference, as described
                  in the project spec, is given by the function roadtrip.total_preference()
            
            :param trip: trip to calculate utility for
            :param node: node to include as well
            :param edge: edge to include as well
            :return: total preference of the trip including the new node and edge
        """
        preference = trip.total_preference()
        
        if not trip.hasNode(node):
            preference += node.preference
        
        if not trip.hasEdge(edge):
            preference += edge.preference
            
        return preference
        

    def find_NodeB(self, edge):
        """
            Find the node associated with location B on an edge

            :param edge: Edge to check
            :return: Node in location B of edge
        """
        for node in self.NodeList:
            if edge.locationB == node.name:
                return node

    def find_NodeA(self, edge):
        """
            Find the node associated with location A on an edge

            :param edge: Edge to check
            :return: Node in location A of edge
        """

        for node in self.NodeList:
            if edge.locationA == node.name:
                return node


def RoundTripRoadTrip(startLoc, LocFile, EdgeFile, ThemeFile, maxTime, x_mph, resultFile, max_trials, forbidden_locations, required_locations, tree):
    """
        Perform a round-trip road trip optimization using the A* search algorithm.

        :param startLoc: Starting location for the road trip.
        :param LocFile: File path containing location data (CSV format).
        :param EdgeFile: File path containing road network data (CSV format).
        :param ThemeFile: File path containing theme data (CSV format)
        :param maxTime: Maximum allowable time for the road trip in minutes.
        :param x_mph: Speed in miles per hour for estimating travel times.
        :param resultFile: File path to save the optimization result.
        :param max_trials: Number of road trips to create and print to user
        :param forbidden_locations: Locations that the user does not want to visit
        :param required_locations: Locations that the user must visit
        :param tree: Which Regression tree to use
    """
    locsAndRoads = Roadtripnetwork(startLoc, LocFile, EdgeFile, ThemeFile, maxTime, x_mph, resultFile, max_trials, forbidden_locations, required_locations)
    locsAndRoads.loadFromFile()
    locsAndRoads.assign_themes()
    locsAndRoads.initializeForSearch(tree)
    locsAndRoads.astar_search()
    return locsAndRoads.solutions, locsAndRoads.themes

def add_suffix(filename, suffix):
    """
        Adds suffix to filename to correctly use file

        :param filename: filename to add to
        :param suffix: suffix to add
        :return: filename with suffix appended
    """

    # Split the filename into base and extension
    base, extension = filename.rsplit('.', 1)

    # Append the suffix and reassemble the filename
    new_filename = f"{base}{suffix}.{extension}"

    return new_filename

def LoadThemesFromFile(attractions_csv_file):
        """
        Load attraction themes from a CSV file into a dictionary.

        Args:
            attractions_csv_file (str): The path to the CSV file containing attraction data.
        
        Returns:
             dict: A dictionary where keys are attraction locations or edges, and values are lists of themes.

        Raises:
            FileNotFoundError: If the specified CSV file does not exist.
        """
        attraction_theme_mapping = {}
        try:
            with open(attractions_csv_file, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    loc_or_edge = row['Loc or Edge Label']
                    themes = row['Themes'].split(', ')
                    attraction_theme_mapping[loc_or_edge] = themes
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{attractions_csv_file}' does not exist in specified directory.")
        return attraction_theme_mapping


def checkLists(required, forbidden):
    """
        Checks to make sure there are no repeats between required and forbidden lists
        since locations cannot be both required and forbidden
        
        :param required: required locations
        :param forbidden: forbidden locations
        :return: if there is not any overlap between the required and forbidden locations
    """
    
    for place in required:
        for place2 in forbidden:
            if place == place2:
                return False
    return True




def main():
    """
        Run program
    """

    num_trials = 1
    print("Welcome to RoundTrip Recommender! Please enter details about your round trip")
    print("If you do not want to specify any of the entries, just click enter and a default value will be used.")
    start_location = input("Enter the starting location for the road trip: ") or "NashvilleTN"

    no_duplicates = False
    while not no_duplicates:
        required_locations = input("Enter any locations that must be a part of your trip (separated by \", \"). Note that all locations must be valid:") or ""
        required_locations_list = required_locations.split(", ")
        forbidden_locations = input("Enter any locations that you do not want to be a part of your trip (separated by "
                                    "\", \"). Note that all locations must be valid:") or " "
        forbidden_locations_list = forbidden_locations.split(", ")
        no_duplicates = checkLists(required_locations_list, forbidden_locations_list)
        if not no_duplicates:
            print("Same cities found in forbidden and required locations")
            print("Please re-enter cities")

    """
    option for soft forbidden location
    """
    location_file = input(
        "Enter the file path containing location data (CSV format): ") or "Road Network - Locations.csv"
    edge_file = input("Enter the file path containing road network data (CSV format): ") or "Road Network - Edges.csv"
    theme_file = input("Enter the file path containing the possible themes (CSV format): ") or "Road Network - Themes.csv"
    max_time = int(input("Enter the maximum allowable time for the road trip: ") or 750)
    speed_in_mph = int(input("Enter the speed in miles per hour for estimating travel times: ") or 60)
    result_file = input("Enter the file path to save the road trip result: ") or "result.txt"
    max_trials = int(input("Enter the maximum number of road trips you would like to display: ") or 3)
    tree = int(input("Enter the regression tree you would like to use (1/2): ") or 1)
    
    round_trips, themes = RoundTripRoadTrip(start_location, location_file, edge_file, theme_file, max_time, speed_in_mph, result_file, max_trials, forbidden_locations_list, required_locations_list, tree)


    runtimes = []
    preferences = []

    first_trip = round_trips.get()


    first_trip[1].print_result(num_trials, start_location, max_time, speed_in_mph, themes)
    first_trip[1].write_result_to_file(num_trials, start_location, max_time, speed_in_mph, themes, result_file)
    #FIXME ADD GIVE NARRATIVE HERE
    first_trip[1].Give_Narrative(start_location, result_file)
    num_trials += 1

    runtimes.append(first_trip[1].time_search)
    preferences.append(first_trip[1].total_preference())

    while num_trials <= max_trials:
        go_again = input(f"\nDo you want to print your next road trip (printed {num_trials - 1} of {max_trials} trips created)? (yes/no): ").lower()
        if go_again != 'yes':
            break
        else:
            cur_trip = round_trips.get()
            cur_trip[1].print_result(num_trials, start_location, max_time, speed_in_mph, themes)
            cur_trip[1].write_result_to_file(num_trials, start_location, max_time, speed_in_mph, themes, result_file)
            cur_trip[1].Give_Narrative(start_location, result_file)
            # FIXME ADD GIVE NARRATIVE HERE
            num_trials += 1
            runtimes.append(cur_trip[1].time_search)
            preferences.append(cur_trip[1].total_preference())

    average_runtime = sum(runtimes) / len(runtimes)
    average_preference = sum(preferences) / len(preferences)
    max_preference = preferences[0]
    min_preference = preferences[0]

    for val in preferences:
        if val > max_preference:
            max_preference = val
        if val < min_preference:
            min_preference = val

    print("\n\nSummary of Output")
    print(f"Average runtime of searches: {average_runtime}")
    print(f"Maximum trip preference: {max_preference}")
    print(f"Average trip preference: {average_preference}")
    print(f"Minimum trip preference: {min_preference}")

    with open(result_file, 'a', encoding='utf-8') as file:
        file.write("Summary of Output\n")
        file.write(f"Average runtime of searches: {average_runtime}\n")
        file.write(f"Maximum trip preference: {max_preference}\n")
        file.write(f"Average trip preference: {average_preference}\n")
        file.write(f"Minimum trip preference: {min_preference}\n")


if __name__ == '__main__':
    main()

"""
In general, there is no solution for a road trip in which the starting location is not on the list of locations in the
provided csv file. For instance, if you wanted to start your trip in a small town called SolonOH, this will not work
because it is not in the csv file provided. In addition, if the required locations are very far away from start locations,
the search will not find the a trip with the required locaiton because the required location is to far to reach in the
time constratins. Finally, if the number of allotted hours is very small or very large (10 or 100000 hours), 
when the speed is very slow (10mphs) the program will either not find a route, or take a very long time to run. 
In general, the amount of time for each road trip is not strictly under the time limit, but have a range of around +- 
50 hours from the given time alloted for the trip. In addition to this, the amount of time as well as the preference 
for the road trips generally decreases between each suggested trip.

Average runtime of all searches for all test runs: 0.558
Average maximum trip preference for all test runs: 8.191
Average total trip preference for all test runs:   7.851
Average minimum trip preference for all test runs: 7.645
"""
