# MSA_transformer2_a3m.py
input: a3m
output: phi and psi and ss(3-state)
model:model_current_a3m(local) + model_current_a3m(multicom) + model_best_a3m.bkp(multico,)

data:(seq_l <= 300) training-40,000  validation-1,378  test-500
batch_size = 50(multicom) / 5(local)
