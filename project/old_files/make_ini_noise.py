from often_use import *

batch_size = 128
random_axis_num = 1024

# seed = 0
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# for i in tqdm(range(int(1e6))):
#     rand_P_seed = torch.rand(batch_size, 3)
#     rand_S_seed = torch.rand(batch_size, 1)
#     randn_theta_seed = torch.rand(batch_size)
#     randn_axis_idx = np.random.choice(random_axis_num, batch_size)

#     path = f'randn/list0randn_batch128_train/{str(int(i)).zfill(10)}.pickle'
#     noise_list = [rand_P_seed, rand_S_seed, randn_theta_seed, randn_axis_idx]
#     pickle_dump(noise_list, path)

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
for i in tqdm(range(int(1e5))):
    rand_P_seed = torch.rand(batch_size, 3)
    rand_S_seed = torch.rand(batch_size, 1)
    randn_theta_seed = torch.rand(batch_size)
    randn_axis_idx = np.random.choice(random_axis_num, batch_size)

    path = f'randn/list0randn_batch128_val/{str(int(i)).zfill(10)}.pickle'
    noise_list = [rand_P_seed, rand_S_seed, randn_theta_seed, randn_axis_idx]
    pickle_dump(noise_list, path)


batch_size = 32

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
for i in tqdm(range(int(1e5))):
    rand_P_seed = torch.rand(batch_size, 3)
    rand_S_seed = torch.rand(batch_size, 1)
    randn_theta_seed = torch.rand(batch_size)
    randn_axis_idx = np.random.choice(random_axis_num, batch_size)

    path = f'randn/list0randn_batch128_tes/{str(int(i)).zfill(10)}.pickle'
    noise_list = [rand_P_seed, rand_S_seed, randn_theta_seed, randn_axis_idx]
    pickle_dump(noise_list, path)