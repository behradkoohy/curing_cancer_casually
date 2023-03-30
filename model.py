import torch
import numpy as np

# Load the data from the file
data = np.loadtxt('cancer_data.txt', delimiter=',', skiprows=1)

# Split the data into features and attributes
features = torch.from_numpy(data[:, :6])
attributes = torch.from_numpy(data[:, 6:])

# Define the linear neural network
input_size = features.size()[1]
output_size = attributes.size()[1]
model = torch.nn.Sequential(
    torch.nn.Linear(input_size, output_size)
)

# Define the loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(100):
    # Forward pass
    outputs = model(features.float())
    loss = criterion(outputs, attributes.float())
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    if (epoch+1) % 10 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss.item()))

# Save the trained model
torch.save(model.state_dict(), 'linear_nn.pth')
