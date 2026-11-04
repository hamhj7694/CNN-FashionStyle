from pathlib import Path
import torch
from PIL import Image

# 1. 저장된 PyTorch 모델 경로 설정
# model_path = "Model/fashion_mood_model_pytorch.pth" #v1
model_path = "Model/fashion_mood_model_pytorch2.pth" #v2

# 2. PyTorch 모델 구조 정의 및 가중치 불러오기
mood_model = models.convnext_tiny(weights=None)
in_features = mood_model.classifier[2].in_features
mood_model.classifier = nn.Sequential(
    nn.Flatten(1),
    nn.Linear(in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)

# 가중치 파일 로드 및 디바이스 매핑
mood_model.load_state_dict(torch.load(model_path, map_location=device))
mood_model = mood_model.to(device)
mood_model.eval()

# 3. (선택) 모델의 분류 개수 확인
print('클래스 개수:', num_classes)

# 4. 모델의 신경망 구조 확인
print(mood_model)

# 5. 분류명 리스트 정의
class_names=['기타', '레트로', '로맨틱', '리조트', '매니시', '모던', '밀리터리', '섹시', '소피스트케이티드', '스트리트', '스포티', '아방가르드', '오리엔탈', '웨스턴', '젠더리스', '컨트리', '클래식', '키치', '톰보이', '펑크', '페미닌', '프레피', '히피', '힙합']

# =======================================
# 6. 이미지 경로를 주면 예측해주는 함수 정의

def predict_image(img_path):
    # 1. 이미지 RGB 변환
    img = Image.open(img_path).convert('RGB')
    
    # 2. 전처리, 4차원 텐서로 변환
    img_tensor = data_transforms(img).unsqueeze(0).to(device)

    # 3. 모델에게 예측 요청
    mood_model.eval()
    with torch.inference_mode():
        outputs = mood_model(img_tensor)
        prob = torch.softmax(outputs, dim=1)

        confidence, idx = torch.max(prob, dim=1) # 가장 높은 것 점수, 인덱스
        
        pred_mood_name = class_names[idx.item()] # 분류명
        confidence_score = confidence.item()     # 신뢰점수 (확률값)

    print(f'예측: {pred_mood_name} ({confidence_score:.2f})')