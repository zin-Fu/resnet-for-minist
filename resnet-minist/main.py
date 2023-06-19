from train import *
from test import *

device = torch.device(DEVICE)
torch.manual_seed(RANDOM_SEED)

model = resnet18(NUM_CLASSES)
model.to(DEVICE)

# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  如果用adam优化器就不需要再设置学习率调度器了（adam自带）
# p.s用adam效果好的多..但是为了多运用一点优化技巧就手动加优化了😭

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)  # 优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 学习率调度器

print("training on ", device)
trainModel(model=model, optimizer=optimizer, scheduler=scheduler)

test_and_show(model=model)




