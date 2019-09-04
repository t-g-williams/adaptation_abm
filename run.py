from model.model import Model
import model.base_inputs as inputs

# compile the inputs
inp = inputs.compile()

# initialize the model
m = Model(inp)

for t in range(m.T):
    m.step()