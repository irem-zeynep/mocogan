# coding: utf-8

import os
import argparse
import glob
import time
import math
import skvideo.io
import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable
from datetime import datetime

from models import Discriminator_I, Discriminator_V, Generator_I, GRU


parser = argparse.ArgumentParser(description='Start trainning MoCoGAN.....')
parser.add_argument('--cuda', type=int, default=1,
                     help='set -1 when you use cpu')
parser.add_argument('--ngpu', type=int, default=1,
                     help='set the number of gpu you use')
parser.add_argument('--batch-size', type=int, default=3,
                     help='set batch_size, default: 3')
parser.add_argument('--niter', type=int, default=120000,
                     help='set num of iterations, default: 120000')
parser.add_argument('--pre-train', type=int, default=0,
                     help='set order of pre-trained model you want to use')

args       = parser.parse_args()
cuda       = args.cuda
ngpu       = args.ngpu
batch_size = args.batch_size
n_iter     = args.niter
pre_train  = args.pre_train

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
if cuda == True:
    torch.cuda.set_device(0)

''' prepare dataset '''
current_path = os.path.dirname(__file__)
resized_path = os.path.join(current_path, 'resized_data')
files = glob.glob(resized_path+'/*')
videos = [ skvideo.io.vread(file, outputdict={"-pix_fmt": "gray"}) for file in files ]
print(videos[0].shape)
# transpose each video to (nc, n_frames, img_size, img_size), and devide by 255
videos = [ video.transpose(3, 0, 1, 2) / 255.0 for video in videos ]

''' prepare video sampling '''
n_videos = len(videos)
T = 128

# for true video
def trim(video):
    start = np.random.randint(0, video.shape[1] - (T+1))
    end = start + T
    return video[:, start:end, :, :]
   
# for input noises to generate fake videof
# note that noises are trimmed randomly from n_frames to T for efficiency
def trim_noise(noise):
    start = np.random.randint(0, noise.size(1) - (T+1))
    end = start + T
    return noise[:, start:end, :, :, :]

def random_choice():
    X = []
    for _ in range(batch_size):
        video = videos[np.random.randint(0, n_videos-1)]
        video = torch.Tensor(trim(video))
        X.append(video)
    X = torch.stack(X)
    return X

# video length distribution
video_lengths = [video.shape[1] for video in videos]

''' set models '''
img_size = 96
nc = 1
ndf = 64 # from dcgan
ngf = 64
d_E = 10
hidden_size = 100 # guess
d_C = 50
d_M = d_E
nz  = d_C + d_M
criterion = nn.BCELoss()

dis_i = Discriminator_I(nc, ndf, ngpu=ngpu)
dis_v = Discriminator_V(nc, ndf, T=T, ngpu=ngpu)
gen_i = Generator_I(nc, ngf, nz, ngpu=ngpu)
gru = GRU(d_E, hidden_size, gpu=cuda)
gru.initWeight()


''' prepare for train '''
label = torch.FloatTensor()

def timeSince(since):
    now = time.time()
    s = now - since
    d = math.floor(s / ((60**2)*24))
    h = math.floor(s / (60**2)) - d*24
    m = math.floor(s / 60) - h*60 - d*24*60
    s = s - m*60 - h*(60**2) - d*24*(60**2)
    return '%dd %dh %dm %ds' % (d, h, m, s)
    
def formatCurrentTime():
    now = datetime.now() 

    return now.strftime("%m_%d_%Y_%H_%M")
    

if pre_train <= 0:
    renamedModelDir = ('trained_models_{}').format(formatCurrentTime())
    renamedVideoDir = ('generated_videos_{}').format(formatCurrentTime())
    os.rename('trained_models', renamedModelDir)
    os.rename('generated_videos', renamedVideoDir)
    os.mkdir('trained_models')
    os.mkdir('generated_videos')
    

trained_path = os.path.join(current_path, 'trained_models')

def checkpoint(model, optimizer, epoch): 
    modelIndex = pre_train + epoch
    filename = os.path.join(trained_path, '%s_epoch-%d' % (model.__class__.__name__, modelIndex))
    torch.save(model.state_dict(), filename + '.model')
    torch.save(optimizer.state_dict(), filename + '.state')

def save_video(fake_video, epoch):
    videoIndex = pre_train + epoch
    outputdata = fake_video * 255
    outputdata = outputdata.astype(np.uint8)
    dir_path = os.path.join(current_path, 'generated_videos')
    file_path = os.path.join(dir_path, 'fakeVideo_epoch-%d.mp4' % videoIndex)
    skvideo.io.vwrite(file_path, outputdata)


''' adjust to cuda '''
if cuda == True:
    dis_i.cuda()
    dis_v.cuda()
    gen_i.cuda()
    gru.cuda()
    criterion.cuda()
    label = label.cuda()


# setup optimizer
lr = 0.0002
betas=(0.5, 0.999)
optim_Di  = optim.Adam(dis_i.parameters(), lr=lr, betas=betas)
optim_Dv  = optim.Adam(dis_v.parameters(), lr=lr, betas=betas)
optim_Gi  = optim.Adam(gen_i.parameters(), lr=lr, betas=betas)
optim_GRU = optim.Adam(gru.parameters(),   lr=lr, betas=betas)


''' use pre-trained models '''
if pre_train > 0:
    dis_i.load_state_dict(torch.load(trained_path + '/Discriminator_I_epoch-{}.model'.format(pre_train)))
    dis_v.load_state_dict(torch.load(trained_path + '/Discriminator_V_epoch-{}.model'.format(pre_train)))
    gen_i.load_state_dict(torch.load(trained_path + '/Generator_I_epoch-{}.model'.format(pre_train)))
    gru.load_state_dict(torch.load(trained_path + '/GRU_epoch-{}.model'.format(pre_train)))
    optim_Di.load_state_dict(torch.load(trained_path + '/Discriminator_I_epoch-{}.state'.format(pre_train)))
    optim_Dv.load_state_dict(torch.load(trained_path + '/Discriminator_V_epoch-{}.state'.format(pre_train)))
    optim_Gi.load_state_dict(torch.load(trained_path + '/Generator_I_epoch-{}.state'.format(pre_train)))
    optim_GRU.load_state_dict(torch.load(trained_path + '/GRU_epoch-{}.state'.format(pre_train)))
    logFile = open("logs.csv", "a")  # append mode
else:
    fileName = 'logs_{}.csv'.format(formatCurrentTime()).replace('/', '_').replace(':', '_')
    os.rename('logs.csv', fileName)
    logFile = open('logs.csv', "a") 
    img_size = 96
    logFile.write(("T: {}, batch: {}, nc: {}, ngf: {}, ndf: {}, d_C:{}, d_M:{}\n").format(T, batch_size, nc, ngf, ndf, d_C, d_M))
    logFile.write("Image Discriminator Loss, Video Discriminator Loss , Genrator Loss, Image Discriminator Fake Mean, Video Discriminator Fake Mean\n")

''' calc grad of models '''
def train_gi(fake_images):

    return loss_generator

def train_gv(fake_videos):

    
    return loss_generator
    
def train_g(fake_images, fake_videos):
    gen_i.zero_grad()
    gru.zero_grad()
    
    fake_labels_v = dis_v(fake_videos.detach())

    label.resize_(fake_videos.size(0)).fill_(1)
    ones_v = Variable(label)
    
    fake_labels_i = dis_i(fake_images.detach())

    label.resize_(fake_images.size(0)).fill_(1)
    ones_i = Variable(label)
    
    loss_generator = criterion(fake_labels_v, ones_v) + criterion(fake_labels_i, ones_i)
    loss_generator.backward()

    optim_Gi.step()
    optim_GRU.step()

    return loss_generator

def train_d(discriminator, optimizer, real_input, fake_input ):
    discriminator.zero_grad()

    real_labels = discriminator(real_input)
    fake_labels = discriminator(fake_input.detach())

    label.resize_(real_input.size(0)).fill_(1)
    ones = Variable(label)

    label.resize_(fake_input.size(0)).fill_(0)
    zeros = Variable(label)
    
    loss_discriminator = criterion(real_labels, ones) + criterion(fake_labels, zeros)
    
    loss_discriminator.backward()

    optimizer.step()
    return loss_discriminator.data, fake_labels.data.mean()

''' gen input noise for fake video '''
def gen_z(n_frames):
    z_C = Variable(torch.randn(batch_size, d_C))
    #  repeat z_C to (batch_size, n_frames, d_C)
    z_C = z_C.unsqueeze(1).repeat(1, n_frames, 1)
    eps = Variable(torch.randn(batch_size, d_E))
    if cuda == True:
        z_C, eps = z_C.cuda(), eps.cuda()

    gru.initHidden(batch_size)
    # notice that 1st dim of gru outputs is seq_len, 2nd is batch_size
    z_M = gru(eps, n_frames).transpose(1, 0)
    z = torch.cat((z_M, z_C), 2)  # z.size() => (batch_size, n_frames, nz)
    return z.view(batch_size, n_frames, nz, 1, 1)


''' train models '''

start_time = time.time()

for epoch in range(1, n_iter+1):
    ''' prepare real images '''
    # real_videos.size() => (batch_size, nc, T, img_size, img_size)
    real_videos = random_choice()
    if cuda == True:
        real_videos = real_videos.cuda()
    real_videos = Variable(real_videos)
    real_img = real_videos[:, :, np.random.randint(0, T), :, :]

    ''' prepare fake images '''
    # note that n_frames is sampled from video length distribution
    n_frames = video_lengths[np.random.randint(0, n_videos)]
    Z = gen_z(n_frames)  # Z.size() => (batch_size, n_frames, nz, 1, 1)
    # trim => (batch_size, T, nz, 1, 1)
    #Z = trim_noise(Z)
    # generate videos
    Z = Z.contiguous().view(batch_size*n_frames, nz, 1, 1)
    fake_videos = gen_i(Z)
    fake_videos = fake_videos.view(batch_size, n_frames, nc, img_size, img_size)
    # transpose => (batch_size, nc, T, img_size, img_size)
    
    fake_videos = fake_videos.transpose(2, 1)
    
    # img sampling
    fake_img = fake_videos[:, :, np.random.randint(0, n_frames), :, :]
    
    fake_videos = torch.stack([torch.as_tensor(trim(video)) for video in fake_videos])
 
    ''' train discriminators '''
    # video
    err_Dv, Dv_fake_mean = train_d(dis_v, optim_Dv, real_videos, fake_videos)

    # image
    err_Di, Di_fake_mean = train_d(dis_i, optim_Di, real_img, fake_img)
    
    ''' train generators '''
    err_G = train_g(fake_img, fake_videos)

    logFile.write('%.4f,%.4f,%.4f,%.4f,%.4f\n'% (err_Di, err_Dv, err_G, Di_fake_mean, Dv_fake_mean))
    logFile.flush()

    if  epoch % 100 == 0:
        save_video(fake_videos[0].data.cpu().numpy().transpose(1, 2, 3, 0), epoch)

    if  epoch % 100 == 0:
        numberFile = open("lastTrainedNumber.txt", "w")
        numberFile.write(('{}').format(pre_train + epoch))
        numberFile.close()
        checkpoint(dis_i, optim_Di, epoch)
        checkpoint(dis_v, optim_Dv, epoch)
        checkpoint(gen_i, optim_Gi, epoch)
        checkpoint(gru,   optim_GRU, epoch)

logFile.close()
