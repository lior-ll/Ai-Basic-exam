b_map = [
    ['H', 'H', 'H'],
    ['H', 'S', 'H'],
    ['H', 'H', 'H']
]
corona = Covid19(b_map, x=0, police=1, medics=0)
#value_iteration_finite_horizon(corona, 5)
#def value_iteration_finite_horizon(mdp, max_steps=5):
    #values = {}
    #for s in mdp.states:
    #    values[s] = vifh(s, max_steps)     
def vifh(s, step):
    R, T = mdp.R, mdp.T
    val = lambda state: values[state] if state in values else vifh(state, step-1)
    if step == 0:
        if s in values:
            print("It is Possible")
            return values[s]
        else:
            values[s] = R(s)
            return values[s]
    values[s] =  R(s) + max([sum([p * val(s1) for (p, s1) in T(s, a)])
                                        for a in mdp.actions(s)])
    return values[s]
values = {} 
mdp = corona
for s in mdp.states:
    values[s] = vifh(s, 6)
