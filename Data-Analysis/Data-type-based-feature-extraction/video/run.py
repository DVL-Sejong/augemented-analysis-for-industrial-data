import requests
import os
import json

url = 'http://127.0.0.1:5000/api/multimodal/video_caption'
headers = {
    'accept': 'application/json'
}

# 결과 저장용 폴더
os.makedirs('results', exist_ok=True)
fail_log_path = 'results/fail_log.txt'
fail_log = open(fail_log_path, 'w', encoding='utf-8')

success_responses = []
error_count = 0

for i in range(7010, 7401):
    filename = f'video{i}.mp4'

    if not os.path.exists(filename):
        print(f"[SKIP] {filename} 없음")
        fail_log.write(f"{filename}: 파일 없음\n")
        error_count += 1
        continue

    with open(filename, 'rb') as f:
        files = {
            'video': (filename, f, 'video/mp4')
        }

        try:
            response = requests.post(url, headers=headers, files=files)
            print(f"[{filename}] 응답코드: {response.status_code}")
            response.raise_for_status()

            result = response.json()
            result['filename'] = filename  # 어떤 파일 응답인지 함께 저장
            success_responses.append(result)

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")
            fail_log.write(f"{filename}: {e}\n")
            error_count += 1

fail_log.close()

# 하나의 파일로 성공 응답 저장
with open('results/success_responses.json', 'w', encoding='utf-8') as out_file:
    json.dump(success_responses, out_file, ensure_ascii=False, indent=2)

print("\n====================")
print(f"총 에러 발생 개수: {error_count}개")
print("성공 응답 저장: results/success_responses.json")
print("에러 로그 저장: results/fail_log.txt")
print("====================")
