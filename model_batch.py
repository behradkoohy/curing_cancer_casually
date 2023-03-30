import torch
import numpy as np

# Load the data from the file
data = np.loadtxt('cancer.txt', skiprows=1)

# Split the data into features and attributes
features = torch.from_numpy(data[:, :6])
attributes = torch.from_numpy(data[:, 6:])
print("feat",features.shape)
print("FEAT SHAPE: ", features.shape)
# Define the linear neural network
input_size = features.size()[1]
output_size = attributes.size()[1]
model = torch.nn.Sequential(
    torch.nn.Linear(input_size, output_size)
)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model

epochs = 1e1
batch=32

for epoch in range(int(epochs)):
    
    # Forward pass
    outputs = model(features.float())
    loss = criterion(outputs, attributes.float())
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print progress
    # if (epoch+1) % 10 == 0:
    if epoch%500 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()/output_size))

# Save the trained model
torch.save(model.state_dict(), 'linear_nn.pth')
