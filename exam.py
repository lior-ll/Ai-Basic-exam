class Covid19(MDP):
    def __init__(self, initial_state, police=0, medics=0, x=0, gamma=0.9):
        self.DIMENSIONS = (len(initial_state), len(initial_state[0])) 
        self.police = police
        self.medics = medics
        self.cell_states = ['H', 'S']
        if self.police:
            self.cell_states.extend(['Q0', 'Q1', 'Q2'])
        if self.medics:
            self.cell_states.append('I')
        self.x = x
        init = self.dict_state_to_tuple_state(self.list_state_to_dict(initial_state))
        states = self.get_all_states()
        transitions = {}
        reward = {}
        terminals = []
        actlist = [('noa',)] # for the case of no police no medics
        for s in states:
            reward[s] = self.get_state_score(s)
            transitions[s] = {}
            for a in self.get_actions(s):
                transitions[s][a] = self.calculate_T(s, a)
        MDP.__init__(self, init, actlist=actlist,
                     terminals=terminals, transitions = transitions, 
                     reward = reward, states = states, gamma=gamma + 0.01*x)
        
    def list_state_to_dict(self, a_map):
        state = {}
        for i in range(0, self.DIMENSIONS[0]):
            for j in range(0, self.DIMENSIONS[1]):
                state[(i, j)] = a_map[i][j]
        return state
    
    def pad_the_input(self, a_map):
        state = {}
        new_i_dim = self.DIMENSIONS[0] + 2
        new_j_dim = self.DIMENSIONS[1] + 2
        for i in range(0, new_i_dim):
            for j in range(0, new_j_dim):
                if i == 0 or j == 0 or i == new_i_dim - 1 or j == new_j_dim - 1:
                    state[(i, j)] = 'U'
                else:
                    state[(i, j)] = a_map[i - 1][j - 1]
        return state
    
    def state_to_agent(self, state):
        state_as_list = []
        for i in range(self.DIMENSIONS[0]):
            state_as_list.append([]*self.DIMENSIONS[1])
            for j in range(self.DIMENSIONS[1]):
                state_as_list[i].append(state[(i + 1, j + 1)][0])
        return state_as_list
    
    def dict_state_to_tuple_state(self, state):
        state_as_list = []
        for i in range(self.DIMENSIONS[0]):
            state_as_list.append([]*self.DIMENSIONS[1])
            for j in range(self.DIMENSIONS[1]):
                state_as_list[i].append(state[(i , j )][0])
        state_as_tuple = tuple(tuple(row) for row in state_as_list)
        return state_as_tuple
    
    def get_all_states(self):
        flat_len = self.DIMENSIONS[0] * self.DIMENSIONS[1]
        unflatter = lambda flat_state: tuple([flat_state[i:i+self.DIMENSIONS[1]] for i in range(0, len(flat_state), self.DIMENSIONS[1])])  
        return [unflatter(state) for state in product(self.cell_states, repeat=flat_len)]
    
    def get_sick_neigbors(self, state, cor):
        i, j = cor
        if state[(i, j)] != 'H':
            return None # not relevant for not healthy
        return sum(['S' in state[(i - 1, j)],
                 'S' in state[(i + 1, j)],
                 'S' in state[(i, j - 1)],
                 'S' in state[(i, j + 1)]])

    def get_state_score(self, state):
        score = 0
        for i in range(self.DIMENSIONS[0]):
            for j in range(self.DIMENSIONS[1]):
                if 'H' in state[i][j]:
                    score += 1
                elif 'I' in state[i][j]:
                    score += 1    
                elif 'S' in state[i][j]:
                    score -= 1
                elif 'Q' in state[i][j]:
                    score -= 5
        return score
    
    def process_state(self, state):
        healthy = []
        sick = []
        for i in range(self.DIMENSIONS[0]):
            for j in range(self.DIMENSIONS[1]):
                if 'H' in state[i][j]:
                    healthy.append((i, j))
                if 'S' in state[i][j]:
                    sick.append((i, j))
        return healthy, sick
    
    def get_actions(self, state):
        # skip police meidics for now....
        actions = [('noa',)]
        if not self.police and not self.medics:
            return actions
        #healthy, sick = self.process_state(state)
        
    def stochastic_cell_dynamic(self, status, n_sicks=None):
        assert (status != 'H') or (n_sicks is not None) # H cell must come with n_sick information
        f = self.x*0.01
        if 'S' in status:
            p_hil = 0.3+f
            return [(p_hil, 'H'), (1-p_hil, 'S')]
        elif 'H' in status:
            if n_sicks == 0:
                return [(1, 'H')]
            else:                
                p_sic = [0.1+f, 0.3+f, 0.7+f, 0.9+f][n_sicks-1]
                return [(p_sic, 'S'), (1-p_sic, 'H')]
        else:
            assert False
        
    def calculate_T(self, state, action):
        #print(state)
        org_state = state
        state = self.pad_the_input(state)
        # police medics operators - IGNROE
        #apademy spread H -> S and recoverd S -> H
        stochastic_states = {}
        for i in range(1, self.DIMENSIONS[0] + 1):
            for j in range(1, self.DIMENSIONS[1] + 1): 
                n_sick = self.get_sick_neigbors(state, (i,j))
                stochastic_states[(i-1,j-1)] = self.stochastic_cell_dynamic(state[(i, j)], n_sick)
        
        #for line in self.state_to_agent(state):
        #    print(line)
        #for key, val in stochastic_states.items():
        #    print(key, ":", val)
        #print(len(list( product(*stochastic_state.values()))))
        all_states = []
        cordinates = stochastic_states.keys()
        for stochastic_state in product(*stochastic_states.values()):
            new_state = {}
            p = 1
            for idx, cor in enumerate(cordinates):
                new_state[cor] = stochastic_state[idx][1]
                p *= float(stochastic_state[idx][0])
            
            #print(p,new_state)
            #print("after convert to tuple")
            #print(self.dict_state_to_tuple_state(new_state))
            all_states.append((p, self.dict_state_to_tuple_state(new_state)))               
        return all_states
corona = Covid19(a_map, x=0)
v_pi = value_iteration(corona)
