import torch
import numpy as np

class ResNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.relu1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(hidden_size, hidden_size)
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x1 = self.linear1(x)
        x2 = self.relu1(x1)
        x3 = self.linear2(x2)
        x4 = self.relu2(x3)
        x5 = self.linear3(x4)
        x6 = self.relu3(x5)
        x7 = self.linear4(x6 + x1 + x3 + x5)
        return x7


# Load the data from the file
data = np.loadtxt('cancer.txt', skiprows=1)

# Split the data into features and attributes
features = torch.from_numpy(data[:, :6])
attributes = torch.from_numpy(data[:, 6:])

# Define the linear neural network
input_size = features.size()[1]
output_size = attributes.size()[1]
model = ResNet(input_size, 128, output_size)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

for epoch in range(5000):
    # Forward pass
    outputs = model(features.float())
    loss = criterion(outputs, attributes.float())
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    # if (epoch+1) % 10 == 0:
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss.item()/output_size))

# Save the trained model
torch.save(model.state_dict(), 'linear_nn.pth')
weights = model[0].weight.detach().numpy()

# for i in range(output_size):
#     print('Top 3 outputs for output vector {}:'.format(i+1))
#     sorted_outputs = np.sort(outputs[:,i].detach().numpy())[::-1]
#     top_outputs = sorted_outputs[:3]
#     for j, output in enumerate(top_outputs):
#         index = np.where(outputs[:,i].detach().numpy() == output)[0][0]
#         feature_values = ', '.join(['{}={:.2f}'.format(f+1, features[index][f].item()) for f in range(input_size)])
#         print('{}. Output value: {:.2f}. Feature weights: {}'.format(j+1, output, ', '.join(['{:.2f}'.format(w) for w in weights[i]])))
#         print('   Features: {}'.format(feature_values))
