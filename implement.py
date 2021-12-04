__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
cfg = {
        'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

dtype=object

class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        output = F.log_softmax(out, dim=1)
        return output

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def baseline_data(num):
          xtrain, ytrain, xtmp,ytmp = get_cifar10()
          x , y = shuffle_list_data(xtrain, ytrain)

          x, y = x[:num], y[:num]
          transform, _ = get_default_data_transforms(train=True, verbose=False)
          loader = torch.utils.data.DataLoader(CustomImageDataset(x, y, transform), batch_size=16, shuffle=True)

          return loader

    def client_update(client_model, optimizer, train_loader, epoch=5):
  
        model.train()
        for e in range(epoch):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = client_model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
        return loss.item()

    def client_syn(client_model, global_model):
 
        client_model.load_state_dict(global_model.state_dict())

    def server_aggregate(global_model, client_models,client_lens):
   
      total = sum(client_lens)
      n = len(client_models)
      global_dict = global_model.state_dict()
      for k in global_dict.keys():
          global_dict[k] = torch.stack([client_models[i].state_dict()[k].float()*(n*client_lens[i]/total) for i in range(len(client_models))], 0).mean(0)
      global_model.load_state_dict(global_dict)
      for model in client_models:
          model.load_state_dict(global_model.state_dict()) 

    def test(global_model, test_loader):
    
      model.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
          for data, target in test_loader:
              data, target = data.cuda(), target.cuda()
              output = global_model(data)
              test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
              pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
              correct += pred.eq(target.view_as(pred)).sum().item()

      test_loss /= len(test_loader.dataset)
      acc = correct / len(test_loader.dataset)

      print('comm-round: {} | average train loss {%0.3g} | test loss {%0.3g} | test acc: {%0.3f}' % (r, loss_retrain / num_selected, test_loss, acc))
      

      return test_loss, acc

classes_pc = 2
num_clients = 20
num_selected = 6
num_rounds = 5
epochs = 2
batch_size = 32
baseline_num = 100
retrain_epochs = 2
      
#### global model ##########
global_model =  VGG('VGG19').cuda()

############# client models ###############################
client_models = [ VGG('VGG19').cuda() for _ in range(num_selected)]
for model in client_models:
    model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global modle 

#include Devise::TestHelpers, type: :controller
  


###### optimizers ################
opt = [optim.SGD(model.parameters(), lr=0.01) for model in client_models]

def baseline_data(num):
          xtrain, ytrain, xtmp,ytmp = get_cifar10()
          x , y = shuffle_list_data(xtrain, ytrain)

          x, y = x[:num], y[:num]
          transform, _ = get_default_data_transforms(train=True, verbose=False)
          loader = torch.utils.data.DataLoader(CustomImageDataset(x, y, transform), batch_size=16, shuffle=True)

          return loader

####### baseline data ############
loader_fixed = baseline_data(baseline_num)

train_loader, test_loader = get_data_loaders(classes_pc=classes_pc, nclients= num_clients,
                                                      batch_size=batch_size,verbose=True)

losses_train = []
losses_test = []
acc_test = []
losses_retrain=[]

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

def client_syn(client_model, global_model):
 
    client_model.load_state_dict(global_model.state_dict())

def client_update(client_model, optimizer, train_loader, epoch=5):
  
        model.train()
        for e in range(epoch):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = client_model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
        return loss.item()

def test(global_model, test_loader):
    
      model.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
          for data, target in test_loader:
              data, target = data.cuda(), target.cuda()
              output = global_model(data)
              test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
              pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
              correct += pred.eq(target.view_as(pred)).sum().item()

      test_loss /= len(test_loader.dataset)
      acc = correct / len(test_loader.dataset)

      print('comm-round: {} | average train loss {%0.3g} | test loss {%0.3g} | test acc: {%0.3f}' % (r, loss_retrain / num_selected, test_loss, acc))
      

      return test_loss, acc


# Runnining FL
for r in range(num_rounds):    #Communication round
    
    # select random clients
    client_idx = np.random.permutation(num_clients)[:num_selected]
    client_lens = [len(train_loader[idx]) for idx in client_idx]


    # client update
    loss = 0
    for i in tqdm(range(num_selected)):
      client_syn(client_models[i], global_model)
      loss += client_update(client_models[i], opt[i], train_loader[client_idx[i]], epochs)
      losses_train.append(loss)

    # server aggregate
    #### retraining on the global server
    loss_retrain =0
    for i in tqdm(range(num_selected)):
      loss_retrain+= client_update(client_models[i], opt[i], loader_fixed)
    losses_retrain.append(loss_retrain)
    
    def test(global_model, test_loader):
    
      model.eval()
      test_loss = 0
      correct = 0
      with torch.no_grad():
          for data, target in test_loader:
              data, target = data.cuda(), target.cuda()
              output = global_model(data)
              test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
              pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
              correct += pred.eq(target.view_as(pred)).sum().item()

      test_loss /= len(test_loader.dataset)
      acc = correct / len(test_loader.dataset)
      return test_loss, acc

    def server_aggregate(global_model, client_models,client_lens):
      test_loss, acc = test(global_model, test_loader)
      losses_test.append(test_loss)
      acc_test.append(acc)

    print('%d-th round' % r)
    #print('average train loss %0.3g | test loss %0.3g | test acc: %0.3f' % (loss_retrain / num_selected, test_loss, acc))
 
  
