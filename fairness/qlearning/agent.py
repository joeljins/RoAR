import json
import os
import random

from .state import State


class Q_State(State):
    '''Augments the game state with Q-learning information'''

    def __init__(self, string):
        super().__init__(string)

        # key stores the state's key string (see notes in _compute_key())
        self.key = self._compute_key()

    def _compute_key(self):
        '''
        Returns a key used to index this state.

        The key should reduce the entire game state to something much smaller
        that can be used for learning. When implementing a Q table as a
        dictionary, this key is used for accessing the Q values for this
        state within the dictionary.
        '''

        # this simple key uses the 3 object characters above the frog
        # and combines them into a key string
        return ''.join([
            self.get(self.frog_x - 1, self.frog_y - 1) or '_',
            self.get(self.frog_x, self.frog_y - 1) or '_',
            self.get(self.frog_x + 1, self.frog_y - 1) or '_',
        ])

    def reward(self):
        '''Returns a reward value for the state.'''

        if self.at_goal:
            return self.score
        elif self.is_done:
            return -10
        else:
            return 0


class Agent:

    def __init__(self, train=None):

        # train is either a string denoting the name of the saved
        # Q-table file, or None if running without training
        self.train = train

        self.previous_state = None
        self.previous_action = None

        # q is the dictionary representing the Q-table
        self.q = {}

        # name is the Q-table filename
        # (you likely don't need to use or change this)
        self.name = train or 'q'

        # path is the path to the Q-table file
        # (you likely don't need to use or change this)
        self.path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'train', self.name + '.json')

        self.load()

    def load(self):
        '''Loads the Q-table from the JSON file'''
        try:
            with open(self.path, 'r') as f:
                self.q = json.load(f)
            if self.train:
                print('Training {}'.format(self.path))
            else:
                print('Loaded {}'.format(self.path))
        except IOError:
            if self.train:
                print('Training {}'.format(self.path))
            else:
                raise Exception('File does not exist: {}'.format(self.path))
        return self

    def save(self):
        '''Saves the Q-table to the JSON file'''
        with open(self.path, 'w') as f:
            json.dump(self.q, f)
        return self

    def choose_action(self, state_string):
        '''
        Returns the action to perform.

        This is the main method that interacts with the game interface:
        given a state string, it should return the action to be taken
        by the agent.

        The initial implementation of this method is simply a random
        choice among the possible actions. You will need to augment
        the code to implement Q-learning within the agent.
        '''

        '''
        return random.choice(State.ACTIONS)
        '''
        alpha = 0.2                     # Learning factor
        gamma = 0.95                    # Discount factor
        exploration_prob = 0.2

        current_state = Q_State( state_string )
        current_key = current_state.key
        
        if current_key not in self.q:
            self.q[ current_key ] = { action: 0 for action in State.ACTIONS }
                                
        if random.random() < exploration_prob:
            action = random.choice(State.ACTIONS)
        else:
            action = max( self.q[ current_key ], key=self.q[ current_key ].get ) 

        if self.previous_state == None:
            pass
        else:
            self.q[ self.previous_key ][ self.previous_action ] = ( ( 1 - alpha ) * 
                self.q[ self.previous_key ][ self.previous_action ] + alpha * ( current_state.reward()
                + gamma * max( self.q[ current_key ].values() ) )
            )
        
        self.previous_state = current_state
        self.previous_key = self.previous_state.key
        self.previous_action = action
        exploration_prob *= 0.99
        self.save()

        return action
        

        
