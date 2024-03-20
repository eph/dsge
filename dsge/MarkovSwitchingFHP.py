class MarkovSwitchingFHP:

    def __init__(self, model1, model2, transition_matrix)
        self.model1 = model1
        self.model2 = model2
        self.transition_matrix = transition_matrix

        # check that the states adn parameters are the same
        assert model1.state_names == model2.state_names
        assert model1.parameter_names == model2.parameter_names

        self.max_k = max(model1.k, model2.k)

        return self

    def solve_LRE(self, p0, max_k = None):

        if max_k is None:
            max_k = self.max_k

            
