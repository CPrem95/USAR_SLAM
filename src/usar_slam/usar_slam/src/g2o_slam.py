#!/usr/bin/env python
# coding: utf-8

# In[1]:


import g2o as g2opy
import numpy as np
import os
import sys
from src.helper_plotting_functions import plot_slam2d

# # Adjust path if needed
# lib_directory = os.path.join(os.getcwd(), "../../build/lib")
# print(f"lib_directory {lib_directory}")

# if not lib_directory in sys.path:
#     sys.path.append(lib_directory)

# import g2opy

# Landmark sensing noise
noise = 0.02


# Import g2o and prepare a GraphSLAM2D class to run an example

# In[2]:


class GraphSLAM2D:
    def __init__(self, verbose=False) -> None:
        '''
        GraphSLAM in 2D with G2O
        '''
        self.optimizer = g2opy.SparseOptimizer()
        self.solver = g2opy.BlockSolverX(g2opy.LinearSolverEigenX())
        self.solver = g2opy.OptimizationAlgorithmLevenberg(self.solver)
        self.optimizer.set_algorithm(self.solver)
        
        # self.optimizer = g2o.SparseOptimizer()
        # self.solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
        # self.algorithm = g2o.OptimizationAlgorithmLevenberg(self.solver)
        # self.optimizer.set_algorithm(self.algorithm)

        self.vertex_count = 0
        self.edge_count = 0
        self.verbose = verbose

    def vertex_pose(self, id):
        '''
        Get position of vertex by id
        '''
        return self.optimizer.vertex(id).estimate()

    def vertex(self, id):
        '''
        Get vertex by id
        '''
        return self.optimizer.vertex(id)

    def edge(self, id):
        '''
        Get edge by id
        '''
        return self.optimizer.edge(id)

    def add_fixed_pose(self, pose, vertex_id=None):
        '''
        Add fixed pose to the graph
        '''
        v_se2 = g2opy.VertexSE2()
        if vertex_id is None:
            vertex_id = self.vertex_count
        v_se2.set_id(vertex_id)
        if self.verbose:
            print("Adding fixed pose vertex with ID", vertex_id)
        v_se2.set_estimate(pose)
        v_se2.set_fixed(True)
        self.optimizer.add_vertex(v_se2)
        self.vertex_count += 1

    def add_odometry(self, northings, eastings, heading, information):
        '''
        Add odometry to the graph
        '''
        # Find the last pose vertex id
        vertices = self.optimizer.vertices()
        if len(vertices) > 0:
            last_id = [v for v in vertices if type(vertices[v]) == g2opy.VertexSE2][0]
            print("Last id is", last_id)
        else:
            raise ValueError("There is no previous pose, have you forgot to add a fixed initial pose?")
        v_se2 = g2opy.VertexSE2()
        if self.verbose:
            print("Adding pose vertex", self.vertex_count)
        v_se2.set_id(self.vertex_count)
        pose = g2opy.SE2(northings, eastings, heading)
        v_se2.set_estimate(pose)
        self.optimizer.add_vertex(v_se2)
        # add edge
        e_se2 = g2opy.EdgeSE2()
        e_se2.set_vertex(0, self.vertex(last_id))
        e_se2.set_vertex(1, self.vertex(self.vertex_count))
        e_se2.set_measurement(pose)
        e_se2.set_information(information)
        self.optimizer.add_edge(e_se2)
        self.vertex_count += 1
        self.edge_count += 1
        if self.verbose:
            print("Adding SE2 edge between", last_id, self.vertex_count-1)

    def add_landmark(self, x, y, information, pose_id, landmark_id=None):
        '''
        Add landmark to the graph
        '''
        relative_measurement = np.array([x, y])
        
        # Check that the pose_id is of type VertexSE2
        if type(self.optimizer.vertex(pose_id)) != g2opy.VertexSE2:
            raise ValueError("The pose_id that you have provided does not correspond to a VertexSE2")
        
        trans0 = self.optimizer.vertex(pose_id).estimate()
        measurement = trans0 * relative_measurement
        
        print(relative_measurement, measurement)
        
        if landmark_id is None:
            landmark_id = self.vertex_count
            v_pointxy = g2opy.VertexPointXY()
            v_pointxy.set_estimate(measurement)
            v_pointxy.set_id(landmark_id)
            if self.verbose:
                print("Adding landmark vertex", landmark_id)
            self.optimizer.add_vertex(v_pointxy)
            self.vertex_count += 1
        # add edge
        e_pointxy = g2opy.EdgeSE2PointXY()
        e_pointxy.set_vertex(0, self.vertex(pose_id))
        e_pointxy.set_vertex(1, self.vertex(landmark_id))
        self.edge_count += 1
        e_pointxy.set_measurement(relative_measurement)
        e_pointxy.set_information(information)
        self.optimizer.add_edge(e_pointxy)
        if self.verbose:
            print("Adding landmark edge between", pose_id, landmark_id)

    def connect_landmarks(self, x, y, information, from_lm_id, to_lm_id):
        '''
        Add measurement between landmarks to the graph
        '''
        relative_measurement = np.array([x, y])
        
        # Check that the pose_id is of type VertexSE2
        # if type(self.optimizer.vertex(from_lm_id)) != g2opy.VertexSE2:
        #     raise ValueError("The pose_id that you have provided does not correspond to a VertexSE2")

        from_id = self.optimizer.vertex(from_lm_id).estimate()
        to_id = self.optimizer.vertex(to_lm_id).estimate()

        print('from:', from_id)
        print('to  :', to_id)

        # add edge
        e_pointxy = g2opy.EdgePointXY()
        e_pointxy.set_vertex(0, self.vertex(from_lm_id))
        e_pointxy.set_vertex(1, self.vertex(to_lm_id))
        self.edge_count += 1
        e_pointxy.set_measurement(relative_measurement)
        e_pointxy.set_information(information)
        self.optimizer.add_edge(e_pointxy)
        if self.verbose:
            print("Connecting landmarks between", from_lm_id, to_lm_id)

    def optimize(self, iterations=10, verbose=None):
        '''
        Optimize the graph
        '''
        self.optimizer.initialize_optimization()
        if verbose is None:
            verbose = self.verbose
        self.optimizer.set_verbose(verbose)
        self.optimizer.optimize(iterations)
        return self.optimizer.chi2()