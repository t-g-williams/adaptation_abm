from model.model import Model
import model.base_inputs as inputs
import plot.single_run as plt
import code

# compile the inputs
inp = inputs.compile()

# initialize the model
m = Model(inp)
# run the model
for t in range(m.T):
    m.step()

# code.interact(local=dict(globals(), **locals()))
# plot
plt.main(m)