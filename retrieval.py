from model import *
from config import *

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
        
    
    if train_flag:
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

    if tsne_flag:
        tsne = manifold.TSNE(n_components=2, metric='cosine', init='pca', random_state=501)
        if dataset == 'imagenet' or dataset == 'coco':
            data_label = np.array([np.argmax(i) for i in data_label])

        X_tsne = tsne.fit_transform(data_predict)
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)  # normalization
        plt.figure(figsize=(8, 8))
        color_list = list(set(mcolors.CSS4_COLORS.keys()) - set(['dimgrey', 'grey', 'darkgrey', 'lightgrey', 'gainsboro', 'whitesmoke', 'white', 'snow', 'darkred', 'mistyrose', 'seashell', 'linen', 'floralwhite', 'ivory', 'honeydew', 'aliceblue', 'azure', 'mintcream', 'ghostwhite', 'lavender', 'lavenderblush', 'lightcyan', 'beige', 'lightyellow', 'lightgoldenrodyellow', 'cornsilk', 'oldlace', 'antiquewhite', 'papayawhip', 'lemonchiffon', 'blanchedalmond']))

        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], '.', color = color_list[data_label[i]], fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.savefig(fig_path, transparent=True, bbox_inches = 'tight', pad_inches = 0)

        return 0

    if dist_flag:
        y_ = np.eye(num_classes)[data_label]    
        y_ = y_.reshape(database_num, num_classes, -1).repeat(num_bits, axis = 2)   
        x_ = data_predict.reshape(database_num, -1, num_bits).repeat(num_classes, axis = 1) 
        proxy = np.sum(x_ * y_, axis = 0) / np.sum(y_, axis = 0)  

        Db = np.sum(np.linalg.norm(proxy[data_label] - data_predict, axis = 1)) / database_num   # calculate intra-distance
        Dw = np.sum(np.sqrt(np.maximum(-2 * np.dot(proxy, proxy.T) + np.sum(np.square(proxy), axis=1, keepdims=True) + np.sum(np.square(proxy), axis=1), np.zeros((num_classes, num_classes))))) / (num_classes * (num_classes - 1))  # calculate inter-distance 

        binary_diff = np.abs(np.sign(data_predict) - data_predict).sum() / (database_num * num_bits)

        print('Db:', Db, ' Dw:', Dw, ' ratio:', Db / Dw, 'diff:', binary_diff)

    data_predict = np.sign(data_predict)
    test_predict = np.sign(test_predict)

    similarity = 1 - np.dot(test_predict, data_predict.T) / num_bits
    sim_ord = np.argsort(similarity, axis=1)

    if figure_flag:
        f = open(method + '_' + str(HHF_flag) + '_' + dataset + '.txt', 'w')
        Mean_precision = np.zeros((10))
        Mean_PR = np.zeros((100))
        for i in range(test_num):
            order=sim_ord[i]
            precision = []
            PR = []
            correct_count = 0
            for j in range(database_num):
                if dataset == 'coco' or dataset == 'imagenet':
                    if np.dot(test_label[i], data_label[order[j]]) > 0:
                        correct_count += 1
                        PR.append(correct_count / (j + 1))
                else:
                    if test_label[i] == data_label[order[j]]:
                        correct_count += 1
                        PR.append(correct_count / (j + 1))
                if j % 100 == 99 and j < 1000:
                    precision.append(correct_count / (j + 1))
            temp = []
            for i in range(100):
                temp.append(PR[round(correct_count * (i + 1) / 100) - 1])

            Mean_precision += np.array(precision)
            Mean_PR += np.array(temp)

        Mean_precision /= test_num
        Mean_PR /= test_num
        f.write(str(Mean_precision) + '\n' + str(Mean_PR))
        return 0

    else:
        apall=np.zeros(test_num)
        for i in range(test_num):
            x=0
            p=0
            order=sim_ord[i]
            for j in range(retrieve):
                if dataset == 'coco' or dataset == 'imagenet':
                    if np.dot(test_label[i], data_label[order[j]]) > 0:
                        x=x+1
                        p=p+float(x)/(j+1)
                else:
                    if test_label[i] == data_label[order[j]]:
                        x=x+1
                        p=p+float(x)/(j+1)
            if p==0:   
                apall[i]=0
            else:
                apall[i]=p/x
        if dataset == 'cifar100-LT' and not train_flag:
            mAP = np.zeros(3)
            for i in range(test_num):
                if test_label[i] < 33:
                    mAP[0] += apall[i]
                elif test_label[i] < 66:
                    mAP[1] += apall[i]
                else:
                    mAP[2] += apall[i]
            mAP = np.divide(mAP, np.array([3300, 3300, 3400]))
        else:
            mAP=np.mean(apall)
            if visual_flag:
                perform_dict = {}
                for i in range(test_num):
                    perform_dict[i] = sim_ord[i][:10]
                performance_order = np.argsort(apall)
                performance_file = open(path+'_performance', 'w')
                if HHF_flag:
                    performance_file.write(str([performance_order[::-1].tolist(), perform_dict]))    
                else:              
                    performance_file.write(str([performance_order.tolist(), perform_dict]))
                performance_file.close()

        return mAP

feature_model = torchvision.models.inception_v3(pretrained = True)
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
                hash_value = feature_model(batch_x)[0]
                batch_x_ = images_.to(device)
                batch_y_ = labels_.to(device)
                hash_value_ = feature_model(batch_x_)[0]

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
                hash_value = feature_model(batch_x)[0]
                loss = model(x = hash_value, batch_y = batch_y, reg = reg_flag)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_time += time.time() - start

                sys.stdout.write('\r')
                sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Time: %.4f'
                    %(epoch , num_epochs, i, total_step, loss.item(), total_time))

                f.write('| Epoch [' + str(epoch) + '/' + str(num_epochs) + '] Iter[' + str(i) + '/' + str(total_step) + '] Loss:' + str(loss.item()) + '\n')

        # f.write('P:\n' + str(P) + '\n')
        scheduler.step()
 
        if epoch > 70:
            mAP = test()
            if mAP > best_map:
                best_map = mAP
                best_epoch = epoch
                print("epoch: ", epoch)
                print("best_map: ", best_map)
                f.write("epoch: " + str(epoch) + '\n')
                f.write("best_map: " + str(best_map) + '\n')
                torch.save(feature_model.state_dict(), model_path)
            else:
                print("epoch: ", epoch)
                print("map: ", mAP)
                print("best_epoch: ", best_epoch)
                print("best_map: ", best_map)
                f.write("epoch: " + str(epoch) + '\n')
                f.write("map: " + str(mAP) + '\n')
                f.write("best_epoch: " + str(best_epoch) + '\n')
                f.write("best_map: " + str(best_map) + '\n')

    f.write("best_map: " + str(best_map) + '\n')
    f.close()

else:
    feature_model.load_state_dict(torch.load(model_path,  map_location = device))
    best_map = test()

print('mAP: ', best_map)
    