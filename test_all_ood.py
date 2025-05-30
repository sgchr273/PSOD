import argparse

import numpy as np
import pandas as pd
import torch
import torchvision as tv
from numpy.linalg import pinv
from scipy.special import softmax

# import extract_utils
import ood_methods
from resnet import ResNet18

from resnet import ResNet34

def parse_args():
    parser = argparse.ArgumentParser(description='Say hello')

    parser.add_argument('--clip_quantile', default=0.99,
                        help='Clip quantile to react')

    parser.add_argument('--img_list', default=None, help='Path to image list')
    parser.add_argument("--id_data", choices=["cifar10", "cifar100", 'imagenet1k'], default="cifar10",
                        help="Which downstream task is ID.")
    parser.add_argument("--ood_data", choices=[ "SVHN", "iSUN", "LSUN", "Places", "Textures", "iNaturalist"], default="SVHN",
                        help="Which downstream task is OOD.")
    parser.add_argument("--cls_size", type=int, default=768,
                        help="size of the class token to be used ")
    parser.add_argument("--model_name",
                        default=["resnet18", "resnet34"],
                        help="Which model to use.")
    parser.add_argument("--model_architecture_type", choices=[ "resnet"],
                        default="resnet",
                        help="what type of model to use")
    parser.add_argument("--base_path", default="./",
                        help="directory where the model is saved.")
    parser.add_argument("--save_path", default="/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/Features",
                        help="directory where the features will be saved.")
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--neco_dim', default=100,
                        help='ETF approximative dimention')

    parser.add_argument("--n_components_null_space", type=int, default=2,
                        help="Number of PCA components to be used for the null space norm")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.id_data == "cifar10":
        num_classes = 10
    elif args.id_data == "cifar100":
        num_classes = 100
    elif args.id_data == "imagenet1k":
        num_classes = 1000

    # if args.model_architecture_type == "resnet":
    #     train_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.id_data}_train.csv'
    #     test_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.id_data}_test.csv'
    #     test_cls_tocken_path_OOD = f'{args.save_path}/{args.model_name}_trained_on_{args.id_data}_OOD_{args.ood_data}_test.csv'

    #     print(f" my args : {args}")
    #     args.ood_features = test_cls_tocken_path_OOD
        # ood_name = args.ood_data
        # print(f"ood datasets: {ood_name}")
    #     model_path = f"{args.base_path}/{args.model_name}_{args.id_data}.pth"

    if args.model_name == 'resnet50':
        model_path = f"{args.base_path}/resnet50_{args.id_data}.pth"
        args.cls_size = 2048

        model = tv.models.resnet50()
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        # model_layers = extract_utils.nested_children(model)
        last_layer = model_layers['fc']
        train_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.id_data}_train.csv'
        test_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.id_data}_test.csv'
        test_cls_tocken_path_OOD = f'{args.save_path}/{args.model_name}_trained_on_{args.id_data}_OOD_{args.ood_data}_test.csv'

    elif args.model_name == "resnet34":
        # model = resnet_models.ResNet34(num_classes)
        # resnet_18_checkpoint = model_path
        # state_dict = torch.load(resnet_18_checkpoint)
        # model.load_state_dict(state_dict['net'], strict=False)
        # print(" acc ", state_dict['acc'])
        # # model_layers = extract_utils.nested_children(model)
        # last_layer = model_layers['linear']
        # args.cls_size = 512
        # model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        model = ResNet34(num_class=num_classes)
        # model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust final layer for 10 classes
        # model.load_state_dict(torch.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/resnet34_cifar10.pth'))
        model.load_state_dict(torch.load(f'/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/model_checkpoints/resnet34_{args.id_data}.pth'))

        train_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.id_data}_train.csv'
        test_cls_tocken_path_ID = f'{args.save_path}/{args.model_name}_ID_{args.id_data}_test.csv'
        test_preds_path_ID = f'{args.save_path}/{args.model_name}_preds_ID_{args.id_data}_test.csv'
        test_cls_tocken_path_OOD = f'{args.save_path}/{args.model_name}_trained_on_{args.id_data}_OOD_{args.ood_data}_test.csv'
        test_preds_path_OOD = f'{args.save_path}/{args.model_name}_preds_trained_on_{args.id_data}_OOD_{args.ood_data}_test.csv'
    elif args.model_name == "resnet18":
        model = ResNet18(num_classes)
        model.load_state_dict(torch.load(f'/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/model_checkpoints/resnet18_{args.id_data}.pth'))
        # model.load_state_dict(torch.load('/usr/local/home/sgchr/Documents/OOD/Multiple_spaces/model.pth'))
        # resnet_18_checkpoint = model_path
        # print(f" model path {model_path}")
        # state_dict = torch.load(resnet_18_checkpoint)
        # model.load_state_dict(state_dict['net'], strict=False)
        # print(" acc ", state_dict['acc'])
        args.cls_size = 512
        # model_layers = extract_utils.nested_children(model)
        # last_layer = model_layers['linear']
        # train_cls_tocken_path_ID = f'{args.save_path}/{args.id_data}_train.npy'
        # test_cls_tocken_path_ID = f'{args.save_path}/{args.id_data}.npy'
        # test_cls_tocken_path_OOD = f'{args.save_path}/{args.ood_data}.npy'
        last_layer = model.linear
        bias = last_layer.bias
        bias.requires_grad = False
        bias = bias.detach().cpu().numpy()
        weight = last_layer.weight
        weight.requires_grad = False
        weight = weight.detach().cpu().numpy()

        # Save the weights and bias as variables
        variables = {'weight': weight, 'bias': bias}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Extract the fully connected layer (fc) weights and bias


    # Specify the filename for the pickle file
    # pickle_filename = 'model_weights_biases.pkl'

    # # Save the variables to the pickle file
    # with open(pickle_filename, 'wb') as f:
    #     pickle.dump(variables, f)
    # print(f'{weight.shape=}, {bias.shape=}')

 ################################################################################################################################################################################################################################################################
    print('load features')
    ID_train_path = f'{args.save_path}/{args.id_data}_{args.model_name}_train.npy'
    train_labels_path = f'{args.save_path}/{args.id_data}_{args.model_name}_train_label.npy'
    # test_labels_path = f'{args.save_path}/{args.id_data}_test_label.npy'
    test_cls_tocken_path_ID = f'{args.save_path}/{args.id_data}_{args.model_name}_test.npy'
    test_cls_tocken_path_OOD = f'{args.save_path}/{args.ood_data}_{args.model_name}_{args.id_data}.npy'
    feature_id_train = np.load(ID_train_path)
    train_labels = np.load(train_labels_path)
    # test_labels = np.load(test_labels_path)
    feature_id_val = np.load(test_cls_tocken_path_ID)
    feature_ood = np.load(test_cls_tocken_path_OOD)
    weight = model.linear.weight
    weight = weight.detach().cpu().numpy()
    bias = model.linear.bias
    bias = bias.detach().cpu().numpy()
    # Print feature shapes for verification
    print(f'{feature_id_train.shape=}, {feature_id_val.shape=}, {feature_ood.shape=}')
    print(f"My args: {args}")

    # Extract weight (w) and bias (b) from the model
    print('Computing logits...')
    logit_id_train = feature_id_train @ weight.T + bias
    logit_id_val = feature_id_val @ weight.T + bias
    logit_ood = feature_ood @ weight.T + bias

    # test_transform = T.Compose([
    # T.Resize((32, 32)),
    # # T.CenterCrop(32),
    # T.ToTensor(),
    # T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    # cifar10_testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    # cifar10_loader = DataLoader(cifar10_testset, batch_size=500, shuffle=True)


    # Compute softmax
    print('Computing softmax...')
    softmax_id_train = softmax(logit_id_train, axis=-1)
    softmax_id_val = softmax(logit_id_val, axis=-1)
    softmax_ood = softmax(logit_ood, axis=-1)

    # Compute u
    u = -np.matmul(pinv(weight), bias)

    # Print results for verification
    print(f'Softmax shapes: {softmax_id_train.shape=}, {softmax_id_val.shape=}, {softmax_ood.shape=}')
    # print(f'Computed u: {u}')
    ood_name = args.ood_data
    print(f"ood datasets: {ood_name}")

    # #---------------------------------------
    method = 'MSP'   
    print(f'\n{method}')
    ood_methods.msp(softmax_id_val, softmax_ood, ood_name)
    # ---------------------------------------
    method = 'MaxLogit'      
    print(f'\n{method}')
    ood_methods.maxLogit(logit_id_val, logit_ood, ood_name)
    #---------------------------------------
    method = 'Energy'   
    print(f'\n{method}')
    ood_methods.energy(logit_id_val, logit_ood, ood_name)
    # ---------------------------------------
    method = 'Energy+React'
    print(f'\n{method}')
    thresh = 0.99
    if args.model_architecture_type == "resnet":
        thresh = 0.9
    clip = np.quantile(feature_id_train, thresh)
    ood_methods.react(feature_id_val, feature_ood, clip, weight, bias, ood_name)
    # ---------------------------------------
    method = 'ViM'  
    print(f'\n{method}')
    ood_methods.vim(feature_id_train, feature_id_val, feature_ood, logit_id_train,
                    logit_id_val, logit_ood, ood_name, args.model_architecture_type, args.model_name, u)

    # ---------------------------------------
    method = 'NECO' 
    print(f'\n{method}')
    ood_methods.neco(feature_id_train, feature_id_val, feature_ood, logit_id_val, logit_ood,
                     model_architecture_type=args.model_architecture_type, neco_dim=args.neco_dim)

    # ---------------------------------------
    method = 'Residual'
    print(f'\n{method}')
    ood_methods.residual(feature_id_train, feature_id_val,
                         feature_ood, args.model_architecture_type, u, ood_name)
    # ---------------------------------------
    method = 'GradNorm'
    print(f'\n{method}')
    ood_methods.gradNorm(feature_id_val, feature_ood,
                         ood_name, num_classes, weight, bias)
    # ---------------------------------------
    method = 'Mahalanobis'
    print(f'\n{method}')
    ood_methods.mahalanobis(feature_id_train, train_labels,
                            feature_id_val, feature_ood, ood_name, num_classes)
    # ---------------------------------------
    method = 'KL-Matching'  
    print(f'\n{method}')
    ood_methods.kl_matching(softmax_id_train, softmax_id_val,
                            softmax_ood, ood_name, num_classes)

if __name__ == '__main__':
    main()


