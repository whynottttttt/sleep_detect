import torch
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from data_loader import eyes_dataset
from model import Net

def accuracy(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

def main():
    PATH = 'weights/classifier_weights_iter_50.pt'

    x_test = np.load('./dataset/x_val.npy').astype(np.float32)  # (288, 26, 34, 1)
    y_test = np.load('./dataset/y_val.npy').astype(np.float32)  # (288, 1)

    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = eyes_dataset(x_test, y_test, transform=test_transform)

    # CUDA 사용 가능 여부 확인 및 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = Net()
    model.to(device)
    model.load_state_dict(torch.load(PATH, map_location=device))
    model.eval()

    total_acc = 0.0
    total_samples = 0

#기존 코드는 각 배치의 정확도를 단순히 더하고 배치의 수로 나눔. 모든 배치의 크기가 동일할 때만 정확
#각 배치의 샘플 수(labels.size(0))를 고려함. 이렇게 하면 마지막 배치의 크기가 다르더라도 정확한 평균을 계산할 수 있음
    with torch.no_grad():
        for data, labels in test_dataloader: #enumerate를 사용하지 않고 for data, labels in test_dataloader:로 간단히 표현
            data, labels = data.to(device), labels.to(device)
            data = data.transpose(1, 3).transpose(2, 3)
            outputs = model(data)
            acc = accuracy(outputs, labels)
            total_acc += acc * labels.size(0) #각 배치의 정확도에 해당 배치의 샘플 수를 곱함. 이는 각 샘플의 정확도 기여도를 올바르게 반영
                                              #count = i를 사용하여 반복 횟수를 세었는데 개선된 코드는 실제 처리된 샘플 수를 사용하므로 이러한 오류를 방지
            total_samples += labels.size(0)  #전체 샘플 수를 누적

    average_acc = total_acc / total_samples # 가중 평균을 계산
    print(f'Average accuracy: {average_acc:.2f}%')

    print('Test finished!')

#if __name__ == '__main__': 구문을 추가하여 스크립트가 직접 실행될 때만 main() 함수가 호출
if __name__ == '__main__':
    main()