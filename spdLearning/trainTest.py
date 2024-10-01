"""Modules for training and testing the neural networks."""


import torch
from tqdm import tqdm

def trainOperation(model, device, dataloader, opti, Loss):
    model.train()
    trainLoss, accuracy, correct, total = 0, 0, 0, 0
    
    for data, target in dataloader:
        target = target.long()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = Loss(output, target)

        opti.zero_grad()
       
        loss.backward()
        opti.step()

        trainLoss += loss.data.item()
        _, prediction = torch.max(output.data, 1)
        total += target.size(0)
        correct += prediction.eq(target.data).cpu().sum().data.item()
    accuracy = 100. * correct/total
    return trainLoss/total, accuracy

def testOperation(model, device, dataloader, Loss):
    with torch.no_grad():
        model.eval()
        testLoss, accuracy, correct, total = 0, 0, 0, 0

        for data, target in dataloader:
            target = target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = Loss(output, target)
            
            testLoss += loss.data.item()
            _, prediction = torch.max(output.data, 1)
            total += target.size(0)
            correct += prediction.eq(target.data).cpu().sum().data.item()
    accuracy = 100. * correct/total
    return testLoss/total, accuracy
