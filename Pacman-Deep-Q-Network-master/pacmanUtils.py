# import pacman game 
from pacman import Directions
from game import Agent
import game
import numpy as np

class PacmanUtils(game.Agent):

    def get_value(self, direction):
        if direction == Directions.NORTH:
            return 0.
        if direction == Directions.EAST:
            return 1.
        if direction == Directions.SOUTH:
            return 2.
        if direction == Directions.WEST:
            return 3.

    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        if value == 1.:
            return Directions.EAST
        if value == 2.:
            return Directions.SOUTH
        if value == 3.:
            return Directions.WEST
			
    def observationFunction(self, state):
        # do observation
        self.terminal = False
        self.observation_step(state)

        return state
		
    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((batch_size, 4))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   

    def mergeStateMatrices(self, stateMatrices):
        """ Merge state matrices to one state tensor """
        stateMatrices = np.swapaxes(stateMatrices, 0, 2)
        total = np.zeros((7, 7))
        for i in range(len(stateMatrices)):
            total += (i + 1) * stateMatrices[i] / 6
        return total

    def getStateMatrices(self, state):

        def get_partial_observation(env, pacman_pos, radius = 1):
            """
            从环境中截取以吃豆人pacman为中心的部分观测区域。

            参数：
            env (numpy.ndarray): 输入的环境张量，形状为 (height, width)。
            pacman_pos (tuple): 吃豆人的位置 (x, y)。
            radius (int): 截取的区域半径。

            返回：
            numpy.ndarray: 截取出的部分观测区域。
            """
            height, width = env.shape
            x, y = pacman_pos
            # 计算截取区域的边界
            x_min = max(-height, x - radius)
            x_max = min(-1, x + radius + 1)
            y_min = max(0, y - radius)
            y_max = min(width, y + radius + 1)
            # print("-height, x - radius\n",-height,x - radius)
            #
            # print("-1, x + radius + 1\n",-1,x + radius + 1)
            #
            # print("0, y - radius\n",0,y - radius)
            #
            # print("width, y + radius + 1\n",width,y + radius + 1)
            # 截取环境张量中的子矩阵

            if x + radius + 1 > -1:
                partial_env = env[x_min: ,y_min:y_max]
            else:
                partial_env = env[x_min:x_max,y_min:y_max]
            # print(partial_env.shape)
            # print(partial_env)
            return partial_env



        """ Return wall, ghosts, food, capsules matrices """ 
        def getWallMatrix(state):
            """ Return matrix with wall coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.layout.walls
            matrix = np.zeros((height, width), dtype=np.int8)
            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            # 获取pacman坐标
            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    new_pos = (-1 - int(pos[1]),int(pos[0]))
            matrix = get_partial_observation(matrix,new_pos)
            return matrix

        def getPacmanMatrix(state):
            """ Return matrix with pacman coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    cell = 1
                    matrix[-1-int(pos[1])][int(pos[0])] = cell

            # 获取pacman坐标
            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    new_pos = (-1 - int(pos[1]),int(pos[0]))
            matrix = get_partial_observation(matrix,new_pos)

            return matrix

        def getGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if not agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            # 获取pacman坐标
            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    new_pos = (-1 - int(pos[1]),int(pos[0]))
            matrix = get_partial_observation(matrix,new_pos)
            return matrix

        def getScaredGhostMatrix(state):
            """ Return matrix with ghost coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            matrix = np.zeros((height, width), dtype=np.int8)

            for agentState in state.data.agentStates:
                if not agentState.isPacman:
                    if agentState.scaredTimer > 0:
                        pos = agentState.configuration.getPosition()
                        cell = 1
                        matrix[-1-int(pos[1])][int(pos[0])] = cell

            # 获取pacman坐标
            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    new_pos = (-1 - int(pos[1]),int(pos[0]))
            matrix = get_partial_observation(matrix,new_pos)
            return matrix

        def getFoodMatrix(state):
            """ Return matrix with food coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            grid = state.data.food
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in range(grid.height):
                for j in range(grid.width):
                    # Put cell vertically reversed in matrix
                    cell = 1 if grid[j][i] else 0
                    matrix[-1-i][j] = cell

            # 获取pacman坐标
            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    new_pos = (-1 - int(pos[1]),int(pos[0]))
            matrix = get_partial_observation(matrix,new_pos)
            return matrix

        def getCapsulesMatrix(state):
            """ Return matrix with capsule coordinates set to 1 """
            width, height = state.data.layout.width, state.data.layout.height
            capsules = state.data.layout.capsules
            matrix = np.zeros((height, width), dtype=np.int8)

            for i in capsules:
                # Insert capsule cells vertically reversed into matrix
                matrix[-1-i[1], i[0]] = 1

            # 获取pacman坐标
            for agentState in state.data.agentStates:
                if agentState.isPacman:
                    pos = agentState.configuration.getPosition()
                    new_pos = (-1 - int(pos[1]),int(pos[0]))
            matrix = get_partial_observation(matrix,new_pos)
            return matrix

        # Create observation matrix as a combination of
        # wall, pacman, ghost, food and capsule matrices
        # width, height = state.data.layout.width, state.data.layout.height 
        width, height = self.width, self.height
        # observation = np.zeros((6, height, width))
        observation = np.zeros((6, 3, 3)) # r=1
        # observation = np.zeros((6, 5, 5)) # r=2
        observation[0] = getWallMatrix(state)
        observation[1] = getPacmanMatrix(state)
        observation[2] = getGhostMatrix(state)
        observation[3] = getScaredGhostMatrix(state)
        observation[4] = getFoodMatrix(state)
        observation[5] = getCapsulesMatrix(state)
        return observation

    def registerInitialState(self, state): # inspects the starting state
        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.episode_reward = 0

        # Reset state
        self.last_state = None
        self.current_state = self.getStateMatrices(state)

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0
        self.episode_number += 1

    def getAction(self, state):
        move = self.getMove(state)

        # check for illegal moves
        legal = state.getLegalActions(0)
        if move not in legal:
            move = Directions.STOP
        return move