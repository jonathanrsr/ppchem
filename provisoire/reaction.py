import sympy
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class Reaction:
    def __init__(self, species, coeffs, rate_law):

        self.species = dict(zip(species, coeffs))
        self.rate_law = sympy.sympify(rate_law)

    def __str__(self):
        return f"Species: {self.species}\n" + \
                f"Rate law: {self.rate_law}"
    
    def reaction_rate(self, concs):
        return self.rate_law.subs({s: c for s, c in zip(self.species, concs)})
    
class Batch():
    def __init__(self, volume, reactions, initial_concs):
        self.volume = volume
        self.reactions = reactions
        self.initial_concs = initial_concs

    def __str__(self):
        reactions_str = "\n".join([f"{i+1}. {reaction}" for i, reaction in enumerate(self.reactions)])
        
        return f"Volume: {self.volume}\n" + \
                f"Réactions: {reactions_str}\n" + \
                f"Concentrations initiales: {self.initial_concs}"
    
    def evolution(self, time, plot = True):
        def ODE(t, y):
            y_dict = dict(zip(self.initial_concs.keys(), y))

            reactions_rates = np.zeros(len(self.reactions))
            transformations_rates = np.zeros(len(y))
            
            for i, reaction in enumerate(self.reactions):
                concs = [y_dict.get(specie, 0) for specie in reaction.species]
                reactions_rates[i] = reaction.reaction_rate(concs)

            for i, specie in enumerate(y_dict):
                for j, reaction in enumerate(self.reactions):
                    if specie in reaction.species:
                        transformations_rates[i] += reactions_rates[j] * reaction.species[specie]

            return transformations_rates
        
        results = solve_ivp(ODE, [0, time], list(self.initial_concs.values()), t_eval=np.linspace(0, time, 1000))

        if plot:
            plt.plot(results.t, results.y.T, linewidth = 1.0)
            plt.legend(list(self.initial_concs.keys()))
            plt.show()