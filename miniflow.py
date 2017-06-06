# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 14:03:37 2017

@author: Sergei Karimov
"""

class Node(object):
    def __init__(self, inbound_nodes = []):
        #Node(s) from which this Node receives values
        self.inbound_nodes = inbound_nodes
        
        #Node(s) to which this Node passes values
        self.outbound_nodes = []
        
        #for each inbound Node here, add this Node as an outbound Node to _that_ Node
        for n in self.outbound_nodes:
            n.outbound_nodes.append(self)
            
        # A calculated value
        self.value = None

    def Forward(self):
        raise NotImplemented

class Input(Node):
    def __init__(self):
        # An Input node has no inbound nodes,
        # so no need to pass anything to the Node instantiator.
        Node.__init__(self)

    # NOTE: Input node is the only node where the value
    # may be passed as an argument to forward().
    #
    # All other node implementations should get the value
    # of the previous node from self.inbound_nodes
    #
    # Example:
    # val0 = self.inbound_nodes[0].value
    def forward(self, value=None):
        # Overwrite the value if one is passed in.
        if value is not None:
            self.value = value

class Add(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        """
        You'll be writing code here in the next quiz!
        """
