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
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

# Train the model

epochs = 5e3
batch=200

for epoch in range(int(epochs)):
    if batch>0:
        for b in range(0,len(data),batch):
            # print("batch length;",len(range(b,b+batch)))
            
                
            # Forward pass
            outputs = model(features.float()[b:b+batch])
            loss = criterion(outputs, attributes.float()[b:b+batch])
                    
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    else:
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
weights = model[0].weight.detach().numpy()
