"""
Program written for MATH 380 project in Fall '21.  The goal of the project is
to find optimal turn strategies in a (simplified) game of Fire Emblem.

This is the slightly cleaned up and better documented version of the program that was
run on the Blugold supercomputing cluster.  The specific version that was run is included
as final_old.py, just in case anything was messed up while cleaning.
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
    '''
    Object to contain the game board and all other relevant data.
    Somewhat overkill now, will probably eventually be replaced by 
    just a 3D numpy array, with 2 dimensions being the board tiles and
    the third dimension being the different stats of the tiles
    '''


    shape = GRID_SIZE
        
    #This will be the game board - actually constructed after units are declared
    grid = np.ndarray([shape[0],shape[1]], dtype=object)
    
    
    def __init__(self, units):
        '''
        Construct the game board.  
        Should pass in a single list of units to place on board.
        All other tiles are assumed to be empty
        '''

        #Initilize all tiles to the zero unit
        #It'd save some memory to have a single, dedicated zero unit 
        ##rather than creating seperate ones, but this works for now
        for i in range(Tiles.shape[0]):
            for j in range(Tiles.shape[1]):
                Tiles.grid[i,j] = Unit((i,j),0,0,0,0)

        #Then write units into their positions
        for unit in units:
            Tiles.grid[unit.POS[0], unit.POS[1]] = unit
            
        return

    
    
    def __str__(self):
        '''Pretty printing of game board
        Mostly for debugging
        '''
        printgrid = np.zeros(Tiles.shape)
        for i in range(Tiles.shape[0]):
            for j in range(Tiles.shape[1]):
                printgrid[i,j] = Tiles.grid[i,j].get_strength()
        return printgrid.__str__()
    

class Unit(Tiles):
    '''
    Class for the units on the board.
    A Unit contains the stats of a unit (HP, HP_MAX, ATK, DEF, and MOV), as well as
    its position and whether or not it is a player unit.

    This class also contains kill method, which will turn a unit into an empty tile,
    and a get_strength method, which will return the unit's power.
    It also contains a move_list method, which will produce a list of every (possibly) valid
    action that the unit can take, given its current position.
    '''
    
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
        
        

def move(grid, unit_pos, move):
    '''
    Function to move units on board.  Technically, it swaps two tiles, but
    it checks that one of those tiles is empty before switching (otherwise
    it raises a ValueError)

    PARAMS:
    -------
    np array grid - The game board
    unit_pos - The position of the unit we wish to move
    move - The move we wish to preform, relative to the units current position
           (i.e if the unit is at (1,2) and we want them at (1,4), we would pass
           in move=(0,2))

    RETURNS:
    --------
    A new np array which is the new game board.  Note that this is a new np array,
    not a modified version of the old one (that caused some weird issues with stuff 
    not being overwritten, for some reason).
    '''
    
    newgrid = np.copy(grid)
    
    #Annoying type conversions
    if type(unit_pos) == list or type(unit_pos) == tuple:
        unit_pos = np.array(unit_pos)
    if type(move) == list or type(move) == tuple:
        move = np.array(move)
    
    #If the new position is beyond the edge of the board, just move
    ##as far as we can
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
    '''
    Function which has one tile attack another.  Takes care of a lot of annoying 
    bookeeping, like making sure we can't attack tiles not on the board or that empty tiles
    can't deal damage.

    PARAMS:
    -------
    np array grid: Game board which we wish to preform the attack on
    np array attacker_pos: Position of the attacker
    np array attack_direc: Direction to attack.  These directions are defined in the Unit class.

    RETURNS:
    --------
    A new np array which is the game board after the attack is preformed.  Same copy warning as in move
    '''

    newgrid = copy.deepcopy(grid)
    
    #Make sure we don't try to attack off the board
    defender_pos = attacker_pos + Unit.directions[attack_direc]
    if (defender_pos[0]<0) or (defender_pos[0] > GRID_SIZE[0]-1) or (defender_pos[1] < 0) or (defender_pos[1] >GRID_SIZE[1]-1):
        return newgrid
    
    
    attacker = newgrid[attacker_pos[0], attacker_pos[1]]
    defender = newgrid[defender_pos[0], defender_pos[1]]
    
    #Exit if either tile is already empty
    if defender.HP <= 0 or attacker.HP <= 0:
        return newgrid
    
    defender_hp = defender.HP - np.max(attacker.ATK - defender.DEF, 0)
    
    attacker_hp = attacker.HP - np.max(defender.ATK - attacker.DEF, 0)
    
    
    #Preform attack in order of attacker, then defender counterattack
    if defender_hp <= 0:
        newgrid[defender.POS[0], defender.POS[1]].kill()
        return newgrid
    
    newgrid[defender.POS[0], defender.POS[1]].HP = defender_hp
    
    if attacker_hp <= 0:
        newgrid[attacker.POS[0], attacker.POS[1]].kill()
        return newgrid
    
    newgrid[attacker.POS[0], attacker.POS[1]].HP = attacker_hp
    
        
    return newgrid

def delta_L(start_state, end_state):
    '''
    Calculate the change in the optimality condition between two game states
    '''
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
    '''
    Function to automatically generate random game board
    '''
    punits = []
    eunits = []
    
    #List of all currently avalible tiles
    positions = [(x,y) for x in range(GRID_SIZE[0]) for y in range(GRID_SIZE[1])]
    
    for i in range(num_players):
        hp = np.random.randint(min_hp, high=max_hp+1)
        atk = np.random.randint(min_atk, high=max_atk+1)
        ddef = np.random.randint(min_def, high=max_def+1)
        move = np.random.randint(min_move, high=max_move+1)
       
        #pop so two units can't get the same tile
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
    '''
    Generates graph of unit actions using a depth first algorithm
    '''

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
    node_dict = {}

    DFS(0, state_stack, punits, unitmoves, graph)

    with open('./graphs/game{}/{}nodes.pickle'.format(game_number, game_perm), 'wb') as f:
        pickle.dump(node_dict, f, protocol=4)
    nx.write_gpickle(graph,'./graphs/game{}/{}.gpickle'.format(game_number, game_perm), protocol=4)
    print('Finished permutation {} of {}'.format(perm_number, total_perms))
