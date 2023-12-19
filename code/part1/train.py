import torch
import numpy as np
import torch.nn.functional as F
import time
import cv2
import config
import  model as m

train_images , train_boxes, train_labels = m.load_data(config.train_dir)
dataset = m.Dataset(train_images, train_labels, train_boxes)

val_images ,val_boxes, val_labels = m.load_data(config.val_dir)
valdataset = m.ValDataset(val_images, val_labels, val_boxes)

dataloader = torch.utils.data.DataLoader(dataset, config.BATCH_SIZE, shuffle=True)
valdataloader = torch.utils.data.DataLoader(valdataset, config.BATCH_SIZE, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(model, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), 0.1, weight_decay=1e-5)
    num_of_epochs = 30
    epochs = []
    losses = []
    for epoch in range(num_of_epochs):
        tot_loss = 0
        tot_correct = 0
        train_start = time.time()
        model.train()
        for batch, (x, y, z) in enumerate(dataloader):
            x, y, z = x.to(device), y.to(device), z.to(device)

            optimizer.zero_grad()
            [y_pred, z_pred] = model(x)

            class_loss = F.cross_entropy(y_pred, y)  # Softmax loss applied
            box_loss = F.mse_loss(z_pred, z)
            box_loss.backward()
            optimizer.step()

            optimizer.step()
            print("Train batch:", batch + 1, " epoch: ", epoch, " ",
                  (time.time() - train_start) / 60, end='\r')

        model.eval()
        for batch, (x, y, z) in enumerate(valdataloader):
            # Converting data from cpu to GPU if available to improve speed
            x, y, z = x.to(device), y.to(device), z.to(device)
            # Sets the gradients of all optimized tensors to zero
            optimizer.zero_grad()
            with torch.no_grad():
                [y_pred, z_pred] = model(x)

                class_loss = F.cross_entropy(y_pred, y)
                box_loss = F.mse_loss(z_pred, z)

            tot_correct += m.get_num_correct(y_pred, y)
            print("Test batch:", batch + 1, " epoch: ", epoch, " ",
                  (time.time() - train_start) / 60, end='\r')
        epochs.append(epoch)
        losses.append(box_loss)
        print("Epoch", epoch, "Accuracy", (tot_correct) / 2.4, "loss:",
              box_loss, " time: ", (time.time() - train_start) / 60, " mins")


def preprocess(img, image_size=256):
    image = cv2.resize(img, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float") / 255.0

    # Expand dimensions as predict expect image in batches
    image = np.expand_dims(image, axis=0)
    return image


def postprocess(image, results):
    # Split the results into class probabilities and box coordinates
    [class_probs, bounding_box] = results

    # First let's get the class label

    # The index of class with the highest confidence is our target class
    class_index = torch.argmax(class_probs)

    # Use this index to get the class name.
    class_label = 0

    # Now you can extract the bounding box too.

    # Get the height and width of the actual image
    h, w = 256, 256

    # Extract the Coordinates
    x1, y1, x2, y2 = bounding_box[0]

    # # Convert the coordinates from relative (i.e. 0-1) to actual values
    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)

    # return the lable and coordinates
    return class_label, (x1, y1, x2, y2), torch.max(class_probs) * 100


def predict(image, scale=0.5):
    model = m.Network()
    model = model.to(device)
    model.eval()
    train(model, config.LEARNING_RATE)

    # # Before we can make a prediction we need to preprocess the image.
    img = cv2.imread(image)
    processed_image = preprocess(img)

    result = model(torch.permute(torch.from_numpy(processed_image).float(), (0, 3, 1, 2)).to(device))

    # After postprocessing, we can easily use our results
    label, (x1, y1, x2, y2), confidence = postprocess(image, result)

    return x1, y1, x2, y2