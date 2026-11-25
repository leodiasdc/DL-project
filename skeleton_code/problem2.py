import load_data
import data

from skeleton_code.skeleton_code.train import set_seed

load_data.download()

split = load_data.load_coco_dominant10()

mean, std = data.channel_stats(split["train"]["images"])        # training set only
train_loader, val_loader, test_loader = data.image_loaders(split, batch_size=64, mean=mean, std=std)

for images, labels in train_loader:                        # (B, 3, 64, 64) float, (B,) long
    

set_seed(0)
model = MyModel(...)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = lambda model, batch: F.cross_entropy(model(batch[0]), batch[1])
history = train(model, optimizer, loss_fn, train_loader,
                val_score_fn=lambda model: my_accuracy(model, val_loader), epochs=20)
print("Number of training examples: ", len(split["train"]["images"]))
print("Number of validation examples: ", len(split["validation"]["images"]))
print("Number of test examples: ", len(split["test"]["images"]))
