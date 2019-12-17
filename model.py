def train_model(model, x_train, y_train, x_val, y_val, optimizer, criterion, epochs=10, verbose=0):
    # Set model to training mode
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()    # Forward pass
        y_pred = model(x_train)    # Compute Loss
        train_loss = criterion(y_pred.squeeze(), y_train)
        train_loss.backward()
        optimizer.step()
        # Prepare for validation
        model.eval()
        y_val_pred = model(x_val)
        val_loss = criterion(y_val_pred.squeeze(), y_val)
        model.train() # Reset to train mode
        if verbose >= 2 and epoch % 100 == 0:
            print('Epoch {}: train loss: {} - val loss: {}'.format(epoch, train_loss.item(), val_loss.item()))    # Backward pass
    model.eval()

