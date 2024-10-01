"""Modules for training and testing the neural networks."""


import torch
from tqdm import tqdm

def trainOperation(model, device, dataloader, cnnOptimizer, rnnOptimizer, Loss):
    model.train()
    trainLoss, accuracy, correct, total = 0, 0, 0, 0
    
    for data, target in dataloader:
        target = target.long()
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = Loss(output, target)

        cnnOptimizer.zero_grad()
        rnnOptimizer.zero_grad()
       
        loss.backward()
        cnnOptimizer.step()
        rnnOptimizer.step()

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
