import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import pandas as pd

from torch.nn import CrossEntropyLoss

# 모델 생성
class Model(nn.Module):
    def __init__(self, in_features=4, h1=8, h2=9, out_features=3):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)
        # 입력에서 은닉층1 -> 은닉층2 -> 출력 까지의 과정을 선형 함수로

    #순전파
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        #x = F.softmax(self.out(x), dim=1)

        return x
    #여기까지 은닉층이 2개인 네트워크 형성(layer는 전부 fully connected layer) - 모든 노드가 다음 레이어의 모든 노드에 연결된 레이어

#모델 객체 생성
torch.manual_seed(13)#난수 고정 함수 // 현재 seed13번 epoch100번이 에러가 가장 적음
model = Model()

X = pd.read_csv('training.csv', header=None)
Y = pd.read_csv('target1.csv', header=None)

X2_test = pd.read_csv('test.csv', header=None)

# 읽어들인 csv파일을 배열로 변환
X = X.values
Y = Y.values
X2_test = X2_test.values

#데이터 텐서화
X_train = torch.FloatTensor(X)
Y_train = torch.LongTensor(Y)
X_test = torch.FloatTensor(X2_test)

#손실함수 정의
#criterion = torch.nn.CrossEntropyLoss()

#최적화 함수 정의
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#훈련 횟수
epochs = 100

#loss를 담을 리스트
losses = []

for i in range(epochs):
    model.train()
    y_pred = model(X_train)
    Y_train = Y_train.squeeze()
    loss = CrossEntropyLoss()(y_pred, Y_train)
    losses.append(loss)

    if i % 10 == 0:
        print(f'epoch {i}, loss is {loss}')

    #역전파 수행
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() #갱신

plt.plot(range(epochs), losses)
plt.ylabel('loss')
plt.xlabel('Epoch')
#plt.show()


# 테스트
correct = 0

print("\n\n\n---------------테스트---------------")
with torch.no_grad():
    for i, data in enumerate(X_test):
        y_val = model.forward(data)

        print(f'{i+1}번째 데이터 결과 : {str(y_val.argmax().item())} 정답 : {Y[i]}')

        if y_val.argmax().item() == Y[i]:
            correct += 1
        else:
            print("error")
print("---------------테스트 결과---------------")
print(f'총 데이터 75개 중 {correct}개 정답')

plt.show()