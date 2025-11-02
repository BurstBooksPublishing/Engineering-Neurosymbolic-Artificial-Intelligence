import numpy as np
import pandas as pd
from causalgraphicalmodels import CausalGraphicalModel

# Define the causal model
pendulum_model = CausalGraphicalModel(
    nodes={"length", "gravity", "period"},
    edges={("length", "period"), ("gravity", "period")}
)

# Visualize the model
pendulum_model.draw()