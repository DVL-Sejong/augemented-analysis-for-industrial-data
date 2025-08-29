import requests
import json

url = 'http://127.0.0.1:5000/api/geolocation/predict/tweet/batch'
headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

# 입력 파일 불러오기
with open("batch_input.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 요청 보내기
response = requests.post(url, headers=headers, json=data)

print(f"📡 Status Code: {response.status_code}")

# 응답 처리
try:
    result = response.json()
    print("📬 Response OK")

    # 결과 파일 저장
    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print("✅ 결과가 result.json 파일로 저장됨")

except Exception as e:
    print("❌ 응답 JSON 파싱 실패:", e)
    print(response.text)
