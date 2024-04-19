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
    
    def evolution(self, time):
        def ODE(t, y):
            y = dict(zip(self.initial_concs.keys(), y))

            reactions_rates = np.zeros(len(self.reactions))
            transformations_rates = np.zeros(len(y))
            transformations_rates_index = 0
            reactions_rates_index = 0

            for reaction in self.reactions:
                concs = np.zeros(len(reaction.species))
                concs_index = 0

                for specie in reaction.species:
                    if specie in y:
                        concs[concs_index] = y[specie]
                        concs_index += 1

                reactions_rates[reactions_rates_index] = reaction.reaction_rate(concs)
                reactions_rates_index += 1

            for specie in y:
                reactions_rates_index = 0

                for reaction in self.reactions:
                    if specie in reaction.species:
                        transformations_rates[transformations_rates_index] += reactions_rates[reactions_rates_index]*reaction.species[specie]
                    reactions_rates_index += 1
                transformations_rates_index += 1

            return transformations_rates

        results = solve_ivp(ODE, [0, time], list(self.initial_concs.values()), t_eval=np.linspace(0, time, 100))

        return results
    
reaction1 = Reaction(['A', 'B', 'C'], [-1, -1, 1], '0.2*A**2*B')
reaction2 = Reaction(['B', 'C', 'D'], [-1, -1, 1], '0.75*B*C')

initial_concs = {'A': 1, 'B': 2, 'C': 0, 'D': 0}

batch = Batch(1, [reaction1, reaction2], initial_concs)

results = batch.evolution(10)
plt.plot(results.t, results.y.T)
plt.show()