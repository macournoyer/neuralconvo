require 'e'
require 'nn'

local mlp = nn.Sequential()
inputs = 2; outputs = 1; HUs = 20; -- parameters
mlp:add(nn.Linear(inputs, HUs))
mlp:add(nn.Tanh())
mlp:add(nn.Linear(HUs, outputs))

local dataset = {
  { torch.Tensor{0, 0}, torch.Tensor{0} },
  { torch.Tensor{0, 1}, torch.Tensor{1} },
  { torch.Tensor{1, 0}, torch.Tensor{1} },
  { torch.Tensor{1, 1}, torch.Tensor{0} },
  size= function() return 4 end
}

local criterion = nn.MSECriterion()
local trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 500
trainer:train(dataset)

print("1 XOR 0 = 1", mlp:forward(torch.Tensor{1, 0}))
print("0 XOR 0 = 0", mlp:forward(torch.Tensor{0, 0}))
print("1 XOR 1 = 0", mlp:forward(torch.Tensor{1, 1}))