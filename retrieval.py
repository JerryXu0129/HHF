from matplotlib.pyplot import axis
from model import *
from config import *
from mpl_toolkits.mplot3d import Axes3D, axes3d
import matplotlib.pyplot as plt
import numpy as np

def prediction(loader):
    for i, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)  
        if i == 0:
            outputs = feature_model(images)
            label = labels
        else:
            outputs = torch.cat((outputs, feature_model(images)), 0)
            label = torch.cat((label, labels), 0)

    if train_flag :
        f.write('output_sample: \n' + str(outputs[:10]) + '\n')
        f.write('label_sample: \n' + str(label[:10]) + '\n')
    else:
        print('output_sample: \n' + str(outputs[:10]))
        print('label_sample: \n' + str(label[:10]))
 
    return outputs.cpu().numpy(), label.cpu().numpy()

# Test the model
def test():
    feature_model.eval()
    if train_flag or not os.path.exists(path + '_data'):

        with torch.no_grad():
            data_predict, data_label = prediction(databaseloader)
            test_predict, test_label = prediction(testloader)
        
        if not train_flag:
            datafile = open(path + '_data', 'w')
            datafile.write(json.dumps([data_predict.tolist(), data_label.tolist(), test_predict.tolist(), test_label.tolist()]))
            datafile.close()
            print('------------- save data -------------')
    else:
        datafile = open(path + '_data', 'r').read()
        data = json.loads(datafile)
        data_predict = np.array(data[0])
        data_label = np.array(data[1])
        test_predict = np.array(data[2])
        test_label = np.array(data[3])
        print('------------- load data -------------')


    data_predict = np.sign(data_predict)
    test_predict = np.sign(test_predict)

    similarity = 1 - np.dot(test_predict, data_predict.T) / num_bits
    sim_ord = np.argsort(similarity, axis=1)

    
    apall=np.zeros(test_num)
    for i in range(test_num):
        x=0
        p=0
        order=sim_ord[i]
        for j in range(retrieve):
            if dataset not in ['cifar10', 'cifar100']:
                if np.dot(test_label[i], data_label[order[j]]) > 0:
                    x += 1
                    p += float(x) / (j + 1)
            else:
                if test_label[i] == data_label[order[j]]:
                    x=x+1
                    p=p+float(x)/(j+1)
        if p > 0:   
            apall[i] = p / x
    mAP=np.mean(apall)
    return mAP
                
if backbone == 'googlenet':
    feature_model = torchvision.models.inception_v3(pretrained = True)
    inchannel = feature_model.fc.in_features
    feature_model.fc = nn.Linear(inchannel, num_bits)
elif backbone == 'resnet':
    feature_model = torchvision.models.resnet50(pretrained = True)
    inchannel = feature_model.fc.in_features
    feature_model.fc = nn.Linear(inchannel, num_bits)
feature_model.to(device)

if train_flag:    # Train the model
    model = HHF().to(device)
    print('------------- model initialized -------------')

    optimizer = torch.optim.SGD([{'params': feature_model.parameters(), 'lr':feature_rate},
                                    {'params': model.parameters(), 'lr':criterion_rate}], momentum = 0.9, weight_decay = 0.0005)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma = 0.5)

    total_step = len(trainloader)
    best_map = 0    #best map
    best_epoch = 0  #best epoch
    total_time = 0

    for epoch in range(num_epochs):
        feature_model.train()

        if based_method =='pair':
            for i, [(images, labels),(images_,labels_)] in enumerate(zip(trainloader,trainloader_)):
                start = time.time()

                batch_x = images.to(device)
                batch_y = labels.to(device)    #   batch_y = 100
                if backbone == 'googlenet':
                    hash_value = feature_model(batch_x)[0]
                else:
                    hash_value = feature_model(batch_x)
                batch_x_ = images_.to(device)
                batch_y_ = labels_.to(device)
                if backbone == 'googlenet':
                    hash_value_ = feature_model(batch_x_)[0]
                else:
                    hash_value_ = feature_model(batch_x_)

                loss = model(x = hash_value, x_ = hash_value_, batch_y = batch_y, batch_y_ = batch_y_, reg = reg_flag)
            
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_time += time.time() - start

                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Time: %.4f'
                    %(epoch , num_epochs, i, total_step, loss.item(), total_time))

                f.write('| Epoch [' + str(epoch) + '/' + str(num_epochs) + '] Iter[' + str(i) + '/' + str(total_step) + '] Loss:' + str(loss.item()) + '\n')
        else:   # proxy-based methods
            for i, (images, labels) in enumerate(trainloader):
                start = time.time()

                batch_x = images.to(device)
                batch_y = labels.to(device)  
                
                if backbone == 'googlenet': 
                    hash_value = feature_model(batch_x)[0]
                else:
                    hash_value = feature_model(batch_x)
                loss = model(x = hash_value, batch_y = batch_y, reg = reg_flag)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_time += time.time() - start

                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Time: %.4f'
                    %(epoch , num_epochs, i, total_step, loss.item(), total_time))
                if datatype != 'toy':
                    f.write('| Epoch [' + str(epoch) + '/' + str(num_epochs) + '] Iter[' + str(i) + '/' + str(total_step) + '] Loss:' + str(loss.item()) + '\n')

        # f.write('P:\n' + str(P) + '\n')
        scheduler.step()


        if epoch > 70:
            mAP = test()
            if mAP > best_map:
                best_map = mAP
                best_epoch = epoch
                print("epoch: ", epoch)
                print("best_" + "mAP: ", best_map)
                f.write("epoch: " + str(epoch) + '\n')
                f.write("best_" + "mAP: " + str(best_map) + '\n')
                torch.save(feature_model.state_dict(), model_path)
            else:
                print("epoch: ", epoch)
                print("mAP: ", mAP)
                print("best_epoch: ", best_epoch)
                print("best_" + "mAP: ", best_map)
                if datatype != 'toy':
                    f.write("epoch: " + str(epoch) + '\n')
                    f.write("mAP: " + str(mAP) + '\n')
                    f.write("best_epoch: " + str(best_epoch) + '\n')
                    f.write("best_" + "mAP: " + str(best_map) + '\n')   
    else:
        f.write("best_" + "mAP: " + str(best_map) + '\n')
        f.close()

else:
    if not os.path.exists(path + '_data'):
        feature_model.load_state_dict(torch.load(model_path,  map_location = device))
    best_map = test()

print("mAP: ", best_map)
    