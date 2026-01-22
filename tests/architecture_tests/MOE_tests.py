from MOE import Moe

num_experts = 4

ex =  nn.ModuleList([nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(4, 2),
            nn.Softmax(dim=-1)
        ) for _ in range(num_experts)])
x = torch.rand(2,4).T
moe = MoE(ex,2,4)
y = moe(x)
print(y, y.shape)