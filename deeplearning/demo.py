from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
class MyDataset(dataset):
    #Read data&preprocess
    def _init_(self,file):
        self.data=
    #Returns one sample atatime
    def _getiten_(self,index):
        return self.data[index]
    #Returns the size of the dataset
    def _len_(self):
        return len(self.data)

#torch.nn - Build your own neural network
class MyModel(nn.Module):
    #initialize  your model & define layers
    def _init_(self):
        super(MyModel, self)._init_()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1)
        )
    #Compute output of your NN
    def forward(self, x):
        return self.net(x)

#load data
    train_data=pd.read_csv('./covid.csv').drop(columns=['data']).values
#preprocessing
    x_train, y_train = train_data[:,:-1], train_data[:,-1]


#**Neural Network Training Setup
    #read data via MyDataset
    dataset =MyDataset(file)
    #put dataset into Dataloader
    tr_set=Dataloader(dataset,batch_size=16,shuffle=True,pin_memory=True)
    #construct model and nove to device(cpu/cuda)
    model =MyModel().to(device)
    #set loss function
    criterion = torch.nn.MSELoss(reduction='mean')
    #set optimizer
    optimizer =torch.optin.SGD(model.paraneters(),0.1)


#Neural Network Training Loop
    #iterate n_cpochs
    for epoch in range(n_epochs):
        #set nodel to train node
        model.train()
        #iterate through the dataloader
        for x,y in tr_set:
            #set gradient to zero
            optimizer.zero_grad()
            #nove dota to device(cpu/cuda)
            x,y=x.to(device),y.to(device)
            #forward pass(conpute output)
            pred=model(x)
            #conpute loss
            loss=criterion(pred,y)
            #conpute gradient (backpropagation)
            loss.backward()
            #update nodel with optinizer
            optimitzer.step()

#**Neural Network Validation Loop
    #set nodel to evaluation node
    model.eval()
    total_loss=0
    #iterate through the dataloader
    for x,y in dv_set:
        #move dota to device(cpu/euda)
        x,y=x.to(device),y.to(device)
        #disable gradient calculation
        with torch.no_grad():
            #forward pass(compute output)
            pred=model(x)
            #conpute loss
            loss=criterion(pred,y)
        #accunulate loss
        total_loss+=loss.cpu().iten()*len(x)
        #compute averaged loss
        avg_loss= total_loss /len(dv_set.dataset)

#**Neural Network Testing Loop
    #set model to evaluation mode
    model.eval()
    preds=[]
    #iterate through the dataloader
    for x in tt_set:
        #move data to device(cpu/cuda)
        x=x.to(device)
        #disable gradient calculation
        with torch.no_grad():
            #forward pass(conpute output)
            pred=model(x)
            #collect predtction
            preds.append(pred.cpu())

#**Save/Load Trained Models
    #save
    torch.save(model.state_dict(),path)
    #load
    ckpt=torch.load(path)
    model.load_state_dict(ckpt)
