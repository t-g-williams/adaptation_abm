from model.model import Model
import model.base_inputs as inputs
import plot.single_run as plt
import code
import time

st1 = time.time()
# compile the inputs
inp = inputs.compile()

# initialize the model
m = Model(inp)
# run the model
for t in range(m.T):
    m.step()

st2 = time.time()
# print(st2-st1)

# code.interact(local=dict(globals(), **locals()))
# plot
plt.main(m)

st3 = time.time()
# print(st3-st2)