# GitHub에 소스 코드 배포 매뉴얼

이 문서는 로컬 프로젝트를 GitHub 리포지토리에 배포하는 과정을 안내합니다.

## 1. .gitignore 파일 생성

버전 관리에서 제외할 파일 및 디렉토리 목록을 `.gitignore` 파일에 정의합니다. 이 과정을 통해 민감한 정보(비밀번호, API 키), 데이터베이스 파일, 로그, 불필요한 시스템 파일 등이 실수로 GitHub에 올라가는 것을 방지합니다.

**예시:**
```
# Environments
.env

# Credentials
credentials.json
token.json

# Databases
*.db
*.sqlite

# Python caches
__pycache__/
```

## 2. Git 저장소 초기화 및 원격 저장소 설정

프로젝트 디렉토리에서 Git을 시작하고 원격 GitHub 리포지토리를 연결합니다.

### Git 초기화
```bash
git init
```

### Git 사용자 정보 설정
커밋을 생성하기 위해 사용자 이름과 이메일을 설정합니다.
```bash
git config --global user.name "your_username"
git config --global user.email "your_email@example.com"
```

### 원격 저장소 연결
GitHub 리포지토리 주소를 `origin`이라는 이름의 원격 저장소로 추가합니다. 보안을 위해 비밀번호 대신 Personal Access Token (PAT) 사용을 권장합니다.

```bash
git remote add origin https://<YOUR_PAT>@github.com/<your_username>/<your_repo>.git
```

만약 이미 원격 저장소가 잘못 설정되어 있다면 URL을 변경합니다.
```bash
git remote set-url origin https://<YOUR_PAT>@github.com/<your_username>/<your_repo>.git
```

## 3. 파일 스테이징 및 커밋

프로젝트의 변경 사항을 로컬 리포지토리에 기록합니다.

```bash
# 모든 변경된 파일을 스테이징
git add .

# 스테이징된 파일들을 커밋
git commit -m "Initial commit"
```

## 4. 브랜치 이름 변경 (master -> main)

최신 Git 규칙과 관례에 따라 기본 브랜치 이름을 `master`에서 `main`으로 변경합니다.

```bash
git branch -m master main
```

## 5. 원격 저장소와 동기화

원격 리포지토리를 생성할 때 README 파일 등을 추가했다면, 로컬 리포지토리와 커밋 히스토리가 달라 푸시가 거부될 수 있습니다. 이때 원격 저장소의 변경 사항을 먼저 가져와 로컬 브랜치와 병합해야 합니다.

```bash
# 원격 저장소의 변경 사항을 가져와서 로컬 브랜치에 재배치(rebase)합니다.
git pull origin main --rebase --allow-unrelated-histories
```

## 6. GitHub에 푸시

로컬 리포지토리의 커밋을 원격 GitHub 리포지토리로 업로드합니다. `-u` 옵션은 로컬 `main` 브랜치가 원격 `origin/main` 브랜치를 추적하도록 설정하여, 다음부터는 `git push`만으로 간단히 푸시할 수 있게 합니다.

```bash
git push -u origin main
```

이제 GitHub 리포지토리에서 업로드된 파일을 확인할 수 있습니다。

## 7. Podman을 이용한 배포 (Ubuntu)

이 섹션에서는 Ubuntu Linux 환경에서 Podman을 사용하여 애플리케이션을 배포하는 방법을 안내합니다.

### 전제 조건

*   Ubuntu Linux 환경
*   Podman 및 `podman-compose` 설치 완료

### 배포 절차

1.  **GitHub에서 소스 코드 복제**

    ```bash
    git clone https://github.com/yblmmen/rag.git
    cd rag
    ```

2.  **Podman을 사용하여 컨테이너 빌드 및 실행**

    `podman-compose`를 사용하여 `docker-compose.yaml` 파일에 정의된 서비스를 빌드하고 실행합니다.

    ```bash
    podman-compose up -d --build
    ```

3.  **애플리케이션 확인**

    배포가 완료되면 웹 브라우저에서 `http://<서버 IP>:80`으로 접속하여 애플리케이션을 확인할 수 있습니다.

4.  **실행 중인 컨테이너 확인**

    ```bash
    podman ps
    ```

5.  **컨테이너 로그 확인**

    ```bash
    podman logs rag_app_container
    ```
