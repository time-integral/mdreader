import mdreader.fmi3 as fmi3

md = fmi3.read_model_description(".tmp/Reference-FMUs-0.0.39/3.0/BouncingBall.fmu")

inputs = []
outputs = []
states = []
parameters = []

for var in md.model_variables:
    if var.causality == "input":
        inputs.append(var)
    elif var.causality == "output":
        outputs.append(var)
    elif var.causality == "state":
        states.append(var)
    elif var.causality == "parameter":
        parameters.append(var)

print(f"Inputs: {[var.name for var in inputs]}")
print(f"Outputs: {[var.name for var in outputs]}")
print(f"States: {[var.name for var in states]}")
print(f"Parameters: {[var.name for var in parameters]}")
