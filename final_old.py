"""
Non-cleaned up version of the program used for MATH 380 project.
This is the version that was actually run on BOSE, though it is significantly
less 'nice' than the cleaned up version in final.py
"""

import numpy as np
import networkx as nx
from itertools import permutations
import copy
import os
import pickle

#Units can generally move about 30% of the map in either direction if no obstacles are present
GRID_SIZE = (12,12)

class Tiles():
    
    shape = GRID_SIZE
        
    #This will be the game board - actually constructed after units are declared
    grid = np.ndarray([shape[0],shape[1]], dtype=object)
    
    
    def __init__(self, units):

        #Initilize all tiles to the zero unit      
        for i in range(Tiles.shape[0]):
            for j in range(Tiles.shape[1]):
                Tiles.grid[i,j] = Unit((i,j),0,0,0,0)

        for unit in units:
            Tiles.grid[unit.POS[0], unit.POS[1]] = unit
            
        return
            
    def attack(self, attacker_pos, defender_pos):
        
        attacker = Tiles.grid[attacker_pos[0], attacker_pos[1]]
        defender = Tiles.grid[defender_pos[0], defender_pos[1]]
        
        if defender.HP or attacker.HP <= 0:
            return
        
        defender_hp = defender.HP - np.max(attacker.ATK - defender.DEF, 0)
        
        attacker_hp = attacker.HP - np.max(defender.ATK - attacker.DEF, 0)
        
        
        if defender_hp <= 0:
            Tiles.grid[defender.POS[0], defender.POS[1]].kill()
            return
        
        #Tiles.grid[defender.POS] = Unit(defender.POS, defender.HP_MAX, defender.ATK, defender.DEF, defender.MOV, HP=defender_hp)
        Tiles.grid[defender.POS[0], defender.POS[1]].HP = defender_hp
        
        if attacker_hp <= 0:
            Tiles.grid[attacker.POS[0], attacker.POS[1]].kill()
            return
        
        #Tiles.grid[attacker.POS] = Unit(attacker.POS, attacker.HP_MAX, attacker.ATK, attacker.DEF, attacker.MOV, HP=attacker_hp)
        Tiles.grid[attacker.POS[0], attacker.POS[1]].HP = attacker_hp
            
        return
    
    def move(self, unit_pos, newpos):
        Tiles.grid[unit_pos[0], unit_pos[1]].POS = newpos
        Tiles.grid[newpos[0], newpos[1]].POS = unit_pos
        
        temp = Tiles.grid[newpos[0], newpos[1]]
        Tiles.grid[newpos[0], newpos[1]] = Tiles.grid[unit_pos[0], unit_pos[1]]
        Tiles.grid[unit_pos[0], unit_pos[1]] = temp


        return
                
    def __str__(self):
        printgrid = np.zeros(Tiles.shape)
        for i in range(Tiles.shape[0]):
            for j in range(Tiles.shape[1]):
                printgrid[i,j] = Tiles.grid[i,j].get_strength()
        return printgrid.__str__()
    

class Unit(Tiles):
    
    
    directions = (np.array([0,0]),np.array([0,1]),np.array([1,0]),np.array([0,-1]),np.array([-1,0]))
    
    def __init__(self, POS, HP_MAX, ATK, DEF, MOV, PLYR=False, HP=None):
        if type(POS) == list or type(POS) == tuple:
            POS = np.array(POS)
        self.POS = POS
        self.HP_MAX = HP_MAX
        if HP == None:
            self.HP = HP_MAX
        else:
            self.HP = HP
        self.ATK = ATK
        self.DEF = DEF
        self.MOV = MOV
        self.PLYR = PLYR
       # move_group = 
        
    def attack(self, direc):
        if direc == 0:
            return
        else:
            super().attack(self.POS, self.POS+Unit.directions[direc])
            return
    
    def move(self, move):
        if type(move) == list or type(move) == tuple:
            move = np.array(move)
        super().move(self.POS, self.POS+move)
        return
        
    def get_strength(self):
        if self.HP_MAX == 0:
            return 0
        else:
            return self.HP/self.HP_MAX * (self.ATK+self.DEF)
        
    def kill(self):
        self.HP_MAX = self.HP = self.ATK = self.DEF = self.MOV = 0
        return
    
    def move_list(self):
        '''returns moves in form (atkdirec, (movex, movey))'''
        movetiles = []
        for i in range(-self.MOV,self.MOV+1):
            for j in range(-self.MOV,self.MOV+1):
                if np.abs(i)+np.abs(j) > self.MOV:
                    continue
                movetiles.append((i,j))
                
        return [(d,mov) for d in range(5) for mov in movetiles]
        
    def __eq__(self, other):
        if isinstance(other, Unit):
            return (self.POS==other.POS) and (self.HP==other.HP) and (self.HP_MAX==other.HP_MAX) and (self.ATK == other.ATK) and (self.DEF == self.DEF) and (self.MOV==other.MOV) and (self.PLYR==other.PLYR)
        else:
            return False
        

def move(grid, unit_pos, move):
    
    newgrid = np.copy(grid)
    
    if type(unit_pos) == list or type(unit_pos) == tuple:
        unit_pos = np.array(unit_pos)
    if type(move) == list or type(move) == tuple:
        move = np.array(move)
    
    newpos = unit_pos + move
    newpos[0] = np.min([newpos[0], GRID_SIZE[0]-1])
    newpos[1] = np.min([newpos[1], GRID_SIZE[1]-1])
    newpos[0] = np.max([0,newpos[0]])
    newpos[1] = np.max([0,newpos[1]])
    
    
    temp = grid[newpos[0], newpos[1]]
    #Disallow moving to a tile with a unit already on it
    if temp.HP != 0:
        raise ValueError("Selected tile already filled")
        return
    newgrid[newpos[0], newpos[1]] = grid[unit_pos[0], unit_pos[1]]
    newgrid[unit_pos[0], unit_pos[1]] = temp
    
    return newgrid

def attack(grid, attacker_pos, attack_direc):
        
    newgrid = copy.deepcopy(grid)
    
    defender_pos = attacker_pos + Unit.directions[attack_direc]
    if (defender_pos[0]<0) or (defender_pos[0] > GRID_SIZE[0]-1) or (defender_pos[1] < 0) or (defender_pos[1] >GRID_SIZE[1]-1):
        return newgrid
    
    
    attacker = newgrid[attacker_pos[0], attacker_pos[1]]
    defender = newgrid[defender_pos[0], defender_pos[1]]
    
    
    if defender.HP <= 0 or attacker.HP <= 0:
        return newgrid
    
    defender_hp = defender.HP - np.max(attacker.ATK - defender.DEF, 0)
    
    attacker_hp = attacker.HP - np.max(defender.ATK - attacker.DEF, 0)
    
    
    if defender_hp <= 0:
        newgrid[defender.POS[0], defender.POS[1]].kill()
        return newgrid
    
    #Tiles.grid[defender.POS] = Unit(defender.POS, defender.HP_MAX, defender.ATK, defender.DEF, defender.MOV, HP=defender_hp)
    newgrid[defender.POS[0], defender.POS[1]].HP = defender_hp
    
    if attacker_hp <= 0:
        newgrid[attacker.POS[0], attacker.POS[1]].kill()
        return newgrid
    
    #Tiles.grid[attacker.POS] = Unit(attacker.POS, attacker.HP_MAX, attacker.ATK, attacker.DEF, attacker.MOV, HP=attacker_hp)
    newgrid[attacker.POS[0], attacker.POS[1]].HP = attacker_hp
    
        
    return newgrid

def delta_L(start_state, end_state):
    start_plyr_strengths = 0
    start_enemy_strengths = 0
    end_plyr_strengths = 0
    end_enemy_strengths = 0
        
    #Iterating over the state array gives the rows, so need a nested loop
    for dummy in start_state:
        for unit in dummy:
            if unit.PLYR:
                start_plyr_strengths+=unit.get_strength()
            else:
                start_enemy_strengths+=unit.get_strength()
    for dummy in end_state:
        for unit in dummy:
            if unit.PLYR:
                end_plyr_strengths+=unit.get_strength()
            else:
                end_enemy_strengths+=unit.get_strength()
            
    start_loss = start_enemy_strengths/start_plyr_strengths
    end_loss = end_enemy_strengths/end_plyr_strengths
    
    #Want to follow paths with low total loss, so delta L should be negative (or as small positive as possible)
    ##So don't use an absolute value or anything
    return end_loss-start_loss

def construct_units(num_players, num_enemies, 
                    max_move=4, max_atk=12, max_def=12, max_hp=24, 
                    min_move=2, min_atk=4, min_def=4, min_hp=8):
    punits = []
    eunits = []
    
    positions = [(x,y) for x in range(GRID_SIZE[0]) for y in range(GRID_SIZE[1])]
    
    for i in range(num_players):
        hp = np.random.randint(min_hp, high=max_hp+1)
        atk = np.random.randint(min_atk, high=max_atk+1)
        ddef = np.random.randint(min_def, high=max_def+1)
        move = np.random.randint(min_move, high=max_move+1)
        
        pos = np.random.randint(low=0, high=len(positions))
        pos = positions.pop(pos)
            
        
        punits.append(Unit(pos, hp, atk, ddef, move, PLYR=True))
        
    for k in range(num_enemies):
        hp = np.random.randint(min_hp, high=max_hp+1)
        atk = np.random.randint(min_atk, high=max_atk+1)
        ddef = np.random.randint(min_def, high=max_def+1)
        move = np.random.randint(min_move, high=max_move+1)
        
        pos = np.random.randint(low=0, high=len(positions))
        pos = positions.pop(pos)

        
        eunits.append(Unit(pos, hp, atk, ddef, move))


    Game = Tiles([*punits, *eunits])

    unitmoves = [unit.move_list() for unit in punits]
    
    return Game, punits, eunits, unitmoves

def DFS(unit_index, state_stack, unit_list, unitmoves, graph):
    start_state_index = state_stack[-1][0]
    start_state = state_stack[-1][1]
    unit = unit_list[unit_index]
    for action in unitmoves[unit_index]:
        try:
            newstate = move(start_state, unit.POS, action[1])
            newpos = np.min((unit.POS+action[1], (GRID_SIZE[0]-1, GRID_SIZE[1]-1)), axis=1)
            newstate = attack(newstate, newpos, action[0])
        except ValueError:
            continue

        state_index = nx.number_of_nodes(graph)
        state_stack.append((state_index,newstate))

        weight = -delta_L(start_state,newstate)

        #Attacking an empty tile is equivilent to not attacking at all,
        #so no need to follow both move trees
        if (action[0] != 0) and (weight==0):
            continue

        graph.add_node(state_index)
        graph.add_edge(start_state_index, state_index, weight=weight)
        node_dict[state_index] = action

        if unit_index != len(unitmoves)-1:
            DFS(unit_index+1, state_stack, unit_list, unitmoves, graph)

        state_stack.pop() 


print('Creating game directry')
game_number = os.getenv('SLURM_ARRAY_TASK_ID')#len(os.listdir('./graphs'))
os.mkdir('./graphs/game{}'.format(game_number))


Game, base_punits, base_eunits, base_unitmoves = construct_units(4,4,max_move=4)
base_unit_index = [x for x in range(1,5)]

print('Saving Units')

np.save('./graphs/game{}/playerunits.npy'.format(game_number), np.array(base_punits))
np.save('./graphs/game{}/enemyunits.npy'.format(game_number), np.array(base_eunits))
punit_data = list(zip(base_unit_index, base_punits, base_unitmoves))


print('Beginning Graph Creation')

total_perms = len(list(permutations(punit_data)))

for perm_number,permutation in enumerate(permutations(punit_data)):

    #Create graph and set up the initial node
    graph = nx.DiGraph()


    state_stack = [(0,Game.grid)]

    graph.add_node(0)


    #Set up the unit order for this permutation
    punits = [x[1] for x in permutation]
    unitmoves = [x[2] for x in permutation]
    game_perm = ''.join([str(x[0]) for x in permutation])
#    zeros_list = np.zeros([len(punits)], dtype=np.bool)
    node_dict = {}

    DFS(0, state_stack, punits, unitmoves, graph)

    with open('./graphs/game{}/{}nodes.pickle'.format(game_number, game_perm), 'wb') as f:
        pickle.dump(node_dict, f, protocol=4)
    nx.write_gpickle(graph,'./graphs/game{}/{}.gpickle'.format(game_number, game_perm), protocol=4)
    print('Finished permutation {} of {}'.format(perm_number, total_perms))
