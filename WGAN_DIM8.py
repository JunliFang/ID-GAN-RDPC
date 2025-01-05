import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.parallel
import torch.nn.functional as F
import math
import os
import xlwt
# %matplotlib inline

EPOCH = 8
BATCH_SIZE = 50
LR = 0.0001
DOWNLOAD_MNIST = True
N_TEST_IMG = 10
DIM = 64
CRITIC_ITERS = 5
LAMBDA = 10
OUTPUT_DIM = 784
DIMENSION = 8
print("CUDA Available: ", torch.cuda.is_available())

train_data = torchvision.datasets.MNIST(root='./mnist/',
                                        train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=DOWNLOAD_MNIST, )
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = test_data.test_data[:BATCH_SIZE]
test_y = test_data.test_labels[:BATCH_SIZE]
view_test_data = test_x.view(-1, 28 * 28).type(torch.FloatTensor) / 255.


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(DIMENSION, 4 * 4 * 4 * DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * DIM, 2 * DIM, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * DIM, DIM, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output = self.preprocess(x)
        output = output.view(-1, 4 * DIM, 4, 4)
        # print output.size()
        output = self.block1(output)
        # print output.size()
        output = output[:, :, :7, :7]
        # print output.size()
        output = self.block2(output)
        # print output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        # print output.size()
        # return output
        return output.view(-1, OUTPUT_DIM)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
            nn.ReLU(True),
            nn.Conv2d(DIM, 2 * DIM, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            nn.Conv2d(2 * DIM, 4 * DIM, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.ReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4 * 4 * 4 * DIM, 1)

    def forward(self, input):
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4 * 4 * 4 * DIM)
        out1 = self.output(out)
        return out1.view(-1)


def calc_gradient_penalty(netD, real_data, fake_data):
    # print real_data.size()
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, 28 * 28)
    alpha = alpha.cuda(device)
    real_data = real_data.view(-1, 28 * 28)
    fake_data = fake_data.view(-1, 28 * 28)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    # interpolates = interpolates.view(BATCH_SIZE, 1, 28, 28)
    interpolates = interpolates.cuda(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

netG = Generator().to(device)
netD = Discriminator().to(device)
optimizerG = torch.optim.Adam(netG.parameters(), lr=LR, betas=(0.5, 0.9))
optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.9))

loss_mse = nn.MSELoss()
loss_ce = nn.CrossEntropyLoss()
criterion = nn.BCELoss()

real_label = 1
fake_label = 0

loss_array_G = []
loss_array_D = []
divergency = []

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)
        b_x = Variable(x.view(-1, 28 * 28))  # b_x save images
        b_label = Variable(y)  # save labels
        # print('epoch',epoch,'step',step)
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True

        for iter_d in range(CRITIC_ITERS):
            netD.zero_grad()

            D_real = -1 * netD(b_x)
            D_real = D_real.mean()
            # print(D_real)
            D_real.backward()

            z = torch.randn(BATCH_SIZE, DIMENSION)
            z = z.to(device)
            x_hat = netG(z)
            x_hat = x_hat.detach()

            D_fake = netD(x_hat)
            D_fake = D_fake.mean()
            D_fake.backward()

            gradient_penalty = calc_gradient_penalty(netD, x, x_hat)
            gradient_penalty.backward()
            divergency.append(-float(D_real) - float(D_fake))
            errD = float(D_real) + float(D_fake) + float(gradient_penalty)

            optimizerD.step()
            loss_array_D.append(errD)

        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation

        netG.zero_grad()
        z = torch.randn(BATCH_SIZE, DIMENSION)
        z = z.to(device)
        x_hat = netG(z)
        loss_G = -netD(x_hat)
        loss_G = loss_G.mean()
        loss_array_G.append(float(loss_G))
        loss_G.backward()
        optimizerG.step()
        torch.cuda.empty_cache()

        if step % 100 == 0:

            z = torch.randn(BATCH_SIZE, DIMENSION)
            z = z.to(device)
            generate1 = netG(z)
            generate2 = generate1 / torch.max(generate1) * 255

            f, a = plt.subplots(2, N_TEST_IMG, figsize=(N_TEST_IMG, 2))
            plt.ion()
            print('EPOCH:', epoch, 'step:', step)
            for i in range(N_TEST_IMG):
                a[0][i].imshow(np.reshape(Variable(view_test_data).data.numpy()[i], (28, 28)), cmap='gray');
                a[0][i].set_xticks(());
                a[0][i].set_yticks(())
                a[1][i].clear()
                a[1][i].imshow(np.reshape(generate2.cpu().data.numpy()[i], (28, 28)), cmap='gray')
                a[1][i].set_xticks(());
                a[1][i].set_yticks(())

            plt.draw();
            plt.pause(0.05)

plt.figure(1)
plt.plot(loss_array_D)
plt.xlabel('iteration')
plt.ylabel('loss_D')

plt.figure(2)
plt.plot(loss_array_G)
plt.xlabel('iteration')
plt.ylabel('lossG')

plt.figure(3)
plt.plot(divergency)
plt.xlabel('iteration')
plt.ylabel('Wasserstein estimate')

workbook = xlwt.Workbook(encoding='ascii')
worksheet = workbook.add_sheet('LOSSD')
column = 0
row = 0
for row in range(len(loss_array_D)):
    worksheet.write(row, column, loss_array_D[row])
column = column + 1
row = 0
for row in range(len(loss_array_G)):
    worksheet.write(row, column, loss_array_G[row])
column = column + 1
row = 0
for row in range(len(divergency)):
    worksheet.write(row, column, divergency[row])

torch.save(netG.state_dict(), 'Generator_dimension8')
# torch.save(netD.state_dict(), 'Wasserstein_Discriminator')