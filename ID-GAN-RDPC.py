import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import math
import os
import xlwt
import matlab.engine
import torch.optim as optim
from PIL import Image


print("CUDA Available: ", torch.cuda.is_available())

from mpl_toolkits.mplot3d import Axes3D

# %matplotlib inline

EPOCH = 5
BATCH_SIZE = 50
LR = 0.0001
DOWNLOAD_MNIST = True
N_TEST_IMG = 10
DIM = 64
CRITIC_ITERS = 5
LAMBDA = 10
OUTPUT_DIM = 784
DIMENSION = 64
NOISE_POWER_DB = 0
TEST_POWER_DB = -10
REPEAT = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
eng = matlab.engine.start_matlab()

train_data = torchvision.datasets.MNIST(root='./mnist/',
                                        train=True,
                                        transform=torchvision.transforms.ToTensor(),
                                        download=DOWNLOAD_MNIST, )
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x = test_data.test_data[:BATCH_SIZE]
test_y = test_data.test_labels[:BATCH_SIZE]
view_test_data = test_x.view(-1, 28 * 28).type(torch.FloatTensor) / 255.
test_generate_data = test_data.test_data.view(-1, 28 * 28).type(torch.FloatTensor) / 255.
test_label = test_data.test_labels[:10000]
noise_power = 10 ** (NOISE_POWER_DB / 10)
test_noise_power = 10 ** (TEST_POWER_DB / 10)


class Encoder(nn.Module):
    def __init__(self, dime):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(

            nn.Linear(28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, dime),
            nn.BatchNorm1d(dime),
            nn.Tanh(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        norm_z = torch.zeros(encoded.size(0), encoded.size(1)).cuda()
        for i in range(encoded.size(0)):
            norm_z[i, :] = encoded[i, :] / torch.sqrt(2 * sum(encoded[i, :] ** 2) / encoded.size(1))
        return encoded


class Generator(nn.Module):
    def __init__(self, dime):
        super(Generator, self).__init__()

        preprocess = nn.Sequential(
            nn.Linear(dime, 4 * 4 * 4 * DIM),
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


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),

            nn.Linear(32, 10),
            nn.Tanh()
        )

    def forward(self, x):
        classify = F.softmax(self.classifier(x))
        return classify


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



def Feature_GAN_DIM_PSNR(dim, noise_power_db, theta_mse, theta_ce, theta_gan):
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    # noise_power = 0
    # noise_power = 1 / (10 ** (noise_power_db / 10))

    sigma_t = torch.tensor([10.0], requires_grad=True)
    # print(sigma_t.is_leaf)
    Gaussian_standard = torch.randn(BATCH_SIZE, dim)
    Gaussian_standard = Gaussian_standard .clone().detach()
    # print(Gaussian_standard.is_leaf)

    Gaussian_standard.requires_grad_(False)
    Gaussian_noise = torch.sqrt(sigma_t) * Gaussian_standard
    Gaussian_noise = Gaussian_noise.to(device)
    # print(sigma_t.is_leaf)

    netE = Encoder(dim).to(device)
    netG = Generator(dim).to(device)
    netD = Discriminator().to(device)
    netC = Classifier().to(device)
    netC_test = Classifier().to(device)

    # sigma_t = sigma_t.clone().detach()
    # sigma_t.requires_grad_(True)

    # optimizerR = optim.SGD([sigma_t],lr=0.01)
    optimizerR =torch.optim.Adam([sigma_t], lr=0.01)
    optimizerE = torch.optim.Adam(netE.parameters(), lr=LR)
    netG.load_state_dict(torch.load('Generator_dimension8'))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=LR, betas=(0.5, 0.9))
    netC.load_state_dict(torch.load('Classifier_Network'))
    netC_test.load_state_dict(torch.load('Classifier_Network_test'))

    loss_mse = nn.MSELoss()
    loss_ce = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()

    loss_array_G = []
    loss_array_D = []
    loss_sigma_t =[]
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
            for p in netE.parameters():  # reset requires_grad
                p.requires_grad = False
            for p in netG.parameters():  # reset requires_grad
                p.requires_grad = False
            for p in netC.parameters():
                p.requires_grad = False  # to avoid computation

            for iter_d in range(CRITIC_ITERS):
                netD.zero_grad()

                D_real = -1 * theta_gan * netD(b_x)
                D_real = D_real.mean()
                # print(D_real)
                D_real.backward()

                z = netE(b_x)

                z = z + Gaussian_noise
                Gaussian_noise = Gaussian_noise.detach()
                z = z.to(device)
                z = z.detach()
                x_hat = netG(z)
                x_hat = x_hat.detach()

                D_fake = theta_gan * netD(x_hat)
                D_fake = D_fake.mean()
                D_fake.backward()

                gradient_penalty = theta_gan * calc_gradient_penalty(netD, x, x_hat)
                gradient_penalty.backward()
                divergency.append(-float(D_real) - float(D_fake))

                errD = float(D_real) + float(D_fake) + float(gradient_penalty)

                optimizerD.step()
                loss_array_D.append(errD)

            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            for p in netE.parameters():  # reset requires_grad
                p.requires_grad = True
            for p in netG.parameters():
                p.requires_grad = False  # to avoid computation
            for p in netC.parameters():
                p.requires_grad = False  # to avoid computation

            netE.zero_grad()
            z = netE(b_x)
            Gaussian_noise = torch.sqrt(sigma_t) * Gaussian_standard
            Gaussian_noise = Gaussian_noise.detach()
            Gaussian_noise = Gaussian_noise.to(device)
            z = z + Gaussian_noise
            z = z.to(device)
            x_hat = netG(z)

            loss_E_GAN = -theta_gan * netD(x_hat) + theta_mse * loss_mse(x_hat, b_x)
            loss_E_GAN = loss_E_GAN.mean()
            loss_E_GAN.backward()

            z = netE(b_x)
            Gaussian_noise = torch.sqrt(sigma_t) * Gaussian_standard
            Gaussian_noise = Gaussian_noise.detach()
            Gaussian_noise = Gaussian_noise.to(device)
            z = z + Gaussian_noise
            z = z.to(device)
            x_hat = netG(z)
            c_hat = netC(x_hat)
            lossR = dim * torch.log(1+torch.reciprocal_(Gaussian_noise))/torch.log(torch.tensor(2.0))

            loss_E_class = theta_ce * loss_ce(c_hat, b_label) + lossR
            loss_E_class = loss_E_class.mean()
            loss_E_class.backward()

            optimizerE.step()
            torch.cuda.empty_cache()

            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            for p in netE.parameters():  # reset requires_grad
                p.requires_grad = False
            for p in netG.parameters():
                p.requires_grad = False  # to avoid computation
            for p in netC.parameters():
                p.requires_grad = False  # to avoid computation

            optimizerR.zero_grad()
            z = netE(b_x)
            Gaussian_noise = torch.sqrt(sigma_t) * Gaussian_standard
            Gaussian_noise.requires_grad_(True)
            # print(Gaussian_noise)
            # Gaussian_noise.requires_grad_(True)
            sigma_t.requires_grad_(True)
            Gaussian_noise = Gaussian_noise.to(device)
            z = z + Gaussian_noise
            z = z.to(device)
            x_hat = netG(z)
            c_hat = netC(x_hat)
            # tt=torch.reciprocal_(Gaussian_noise)
            # print(tt)
            lossR = dim * torch.log(1+torch.reciprocal_(torch.abs(Gaussian_noise)))/torch.log(torch.tensor(2.0))
            lossR = lossR.mean()
            # print('lossR',lossR)

            MSE = loss_mse(x_hat, b_x)
            MSE = MSE.mean()
            # print('MSE',MSE)
            #
            CE = loss_ce(c_hat, b_label)
            CE = CE.mean()
            # print('ce',CE)
            #
            # D_real = -1 * theta_gan * netD(b_x)
            # D_real = D_real.mean()
            D_fake = theta_gan * netD(x_hat)
            D_fake = D_fake.mean()
            # gradient_penalty = theta_gan * calc_gradient_penalty(netD, x, x_hat)
            # print('WD',D_real + D_fake + gradient_penalty)
            # loss_sigma = lossR

            loss_sigma = lossR +  theta_mse*MSE + theta_ce*CE  - D_fake
            loss_sigma_t.append(loss_sigma)
            # print(loss_sigma)

            loss_sigma.backward()
            optimizerR.step()
            with torch.no_grad():
                sigma_t.clamp_(min=0.0001)

            if step % 100 == 0:
                print('Epoch:', epoch, 'step:', step, 'sigma_t',sigma_t,'lossR',lossR,'MSE',theta_mse*MSE,'CE',theta_ce*CE)
                # print('standard Gaussian',Gaussian_standard)

    plt.figure()
    plt.plot(loss_array_D)
    plt.xlabel('lossD')
    plt.show()
    # plt.figure()
    # plt.plot(loss_array_G)
    # plt.xlabel('lossG')
    plt.figure()
    plt.plot(loss_sigma_t)
    plt.xlabel('sigma_t')
    plt.show()

    mse_snr = []
    acc_snr = []
    NIQE_snr = []

    rate_bites = [1, 3, 5, 7, 10, 20, 30, 40, 50, 60]
    # rate_bites = [40]
    noise = []
    for k in range(len(rate_bites)):
        noise.append(1 / (2 ** (rate_bites[k] / dim) - 1))

    for index, s in enumerate(noise):
        image_i = 0
        mse_array = []
        acc_array = []
        # NIQE_array = []
        directory = "C:/Users/PS/Desktop/niqe-master/test_imgs/IDGANrate_{}".format(int(rate_bites[index]))
        os.mkdir(directory)
        for count in range(200):
            data = test_generate_data[count * BATCH_SIZE:(count + 1) * BATCH_SIZE]
            data_label = test_label[count * BATCH_SIZE:(count + 1) * BATCH_SIZE]
            data = data.to(device)
            encode1 = netE(data)
            double_tensor = torch.ones(BATCH_SIZE, dim)
            Gaussian_noise = torch.nn.init.normal_(double_tensor, mean=0, std=s ** (1 / 2))
            Gaussian_noise = Gaussian_noise.to(device)
            encode1 = encode1 + Gaussian_noise
            generate1 = netG(encode1).view(-1, 28 * 28)
            classify1 = netC_test(generate1)

            generate2 = netG(encode1).view(BATCH_SIZE, 28, 28, 1)
            split_images = torch.split(generate2, 1, 0)


            for u in range(len(split_images)):
                image = split_images[u].squeeze()
                image = image / torch.max(image) * 255
                image = image.cpu().detach().numpy()

                pil_image = Image.fromarray(np.uint8(image))
                filename = "{}.jpeg".format(image_i)
                filepath = os.path.join(directory, filename)
                pil_image.save(filepath)
                image_i = image_i + 1

            pred_y = torch.max(classify1, 1)[1].data.squeeze()
            accuracy = float(sum(pred_y.cpu().data.numpy() == data_label.cpu().data.numpy())) / float(test_y.size(0))
            acc_array.append(accuracy)

            mse = torch.mean((data / 1. - generate1 / 1.) ** 2)
            mse = mse.cpu().data.numpy()
            mse_array.append(mse)

        mean_mse = np.mean(mse_array)
        mean_acc = np.mean(acc_array)
        # mean_NIQE = np.mean(NIQE_array)

        mse_snr.append(mean_mse)
        acc_snr.append(mean_acc)
        # NIQE_snr.append(mean_NIQE)
        # mean_WD = np.mean(WD_array)
        print('MSE:', mean_mse, 'classification accuracy:', mean_acc)

    return mse_snr, acc_snr


device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
# dim_interval = np.linspace(2,6,5)

dim = [8]
theta_mse = 100000
theta_ce = 1
theta_gan = 1
theta = [1]
train_noise_DB = 5

# for d in range(len(dim_interval)):
#    dim.append(2**(dim_interval[d]))

average_mse_i = []
average_WD_i = []
average_acc_i = []

workbook_mse = xlwt.Workbook(encoding='ascii')
worksheet_mse = workbook_mse.add_sheet('dim_mse_perception')
# workbook_NIQE = xlwt.Workbook(encoding='ascii')
# worksheet_NIQE = workbook_NIQE.add_sheet('dim_mse_perception')
workbook_acc = xlwt.Workbook(encoding='ascii')
worksheet_acc = workbook_acc.add_sheet('dim_mse_perception')

# for d in range(len(dim)):
for t in range(len(theta)):
    for d in range(len(dim)):
        print('dimension', dim, 'noise_db', train_noise_DB, 'theta_mse', theta_mse, 'theta_ce', theta_ce, 'theta_gan',
              theta_gan)
        mse_snr1, acc_snr1 = Feature_GAN_DIM_PSNR(dim[d], train_noise_DB, theta_mse, theta_ce, theta_gan)
        for i in range(len(mse_snr1)):
            worksheet_mse.write(t, i, str(mse_snr1[i]))
        # for i in range(len(NIQE_snr1)):
        #     worksheet_NIQE.write(t, i, str(NIQE_snr1[i]))
        for i in range(len(acc_snr1)):
            worksheet_acc.write(t, i, str(acc_snr1[i]))

workbook_mse.save(
    '250105_N_{var}_dim8_D_{var1}_P_{var2}_C_{var3}_mse.xls'.format(var=train_noise_DB, var1=theta_mse, var2=theta_gan,
                                                                    var3=theta_ce))
# workbook_NIQE.save(
#     '250105_N_{var}_dim8_D_{var1}_P_{var2}_C_{var3}_WD.xls'.format(var=train_noise_DB, var1=theta_mse, var2=theta_gan,
#                                                                    var3=theta_ce))
workbook_acc.save(
    '250105_N_{var}_dim8_D_{var1}_P_{var2}_C_{var3}_acc.xls'.format(var=train_noise_DB, var1=theta_mse, var2=theta_gan,
                                                                    var3=theta_ce))
