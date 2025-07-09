# 🚀 Slack MCP Server

FastMCP를 사용한 완전한 Slack 통합 MCP 서버입니다. Cursor IDE에서 Slack API의 모든 기능을 자연어로 사용할 수 있게 해줍니다.

## 📋 목차

- [특징](#-특징)
- [요구사항](#-요구사항)
- [빠른 설치](#-빠른-설치)
- [수동 설치](#-수동-설치)
- [사용법](#-사용법)
- [사용 가능한 도구](#-사용-가능한-도구)
- [프로젝트 구조](#-프로젝트-구조)
- [개발자 가이드](#-개발자-가이드)
- [문제 해결](#-문제-해결)
- [라이선스](#-라이선스)

## ✨ 특징

### 필수 기능 (과제 요구사항)
- ✅ **메시지 전송**: 채널에 메시지 전송
- ✅ **채널 목록 조회**: 접근 가능한 모든 채널 조회
- ✅ **채널 히스토리 조회**: 지정된 채널의 메시지 히스토리
- ✅ **다이렉트 메시지**: 1:1 메시지 전송

### 추가 기능
- ✅ **사용자 목록 조회**: 워크스페이스 사용자 정보
- ✅ **반응 추가**: 메시지에 이모지 반응 추가
- ✅ **메시지 검색**: 키워드 기반 메시지 검색
- ✅ **연결 테스트**: API 연결 상태 확인

### 기술적 특징
- 🚀 **FastMCP 기반**: 최신 MCP 프레임워크 사용
- ⚡ **비동기 처리**: aiohttp를 이용한 고성능 API 호출
- 🛡️ **완전한 에러 처리**: 상세한 오류 메시지와 복구 가능한 예외 처리
- 📝 **타입 힌트**: 완전한 타입 안정성
- 🧪 **포괄적 테스트**: 자동화된 테스트 스위트
- 📊 **로깅**: 구조화된 로깅 시스템
- 🔧 **자동 설치**: 원클릭 설치 스크립트

## 📋 요구사항

- **Python 3.8+**
- **pip** (패키지 관리자)
- **Cursor IDE** (MCP 클라이언트)
- **Slack Bot Token** (xoxb-로 시작)

### Slack Bot 권한

Slack 앱에 다음 권한이 필요합니다:

```
Bot Token Scopes:
- channels:history    # 채널 메시지 읽기
- channels:read       # 채널 정보 조회
- chat:write          # 메시지 전송
- groups:history      # 비공개 그룹 메시지 읽기
- groups:read         # 비공개 그룹 정보 조회
- im:history          # DM 메시지 읽기
- im:read             # DM 정보 조회
- im:write            # DM 전송
- mpim:history        # 멀티파티 DM 메시지 읽기
- mpim:read           # 멀티파티 DM 정보 조회
- mpim:write          # 멀티파티 DM 전송
- reactions:write     # 반응 추가
- search:read         # 메시지 검색
- users:read          # 사용자 정보 조회
```

## 🚀 빠른 설치

### 1. 자동 설치 (권장)

```bash
cd slack-mcp
python install.py
```

설치 스크립트가 다음을 자동으로 처리합니다:
- 패키지 설치
- 환경 변수 설정
- Cursor MCP 설정 업데이트
- 설치 테스트 실행

### 2. Cursor 재시작

설치 완료 후 Cursor를 재시작하면 Slack MCP 도구를 사용할 수 있습니다.

## 🔧 수동 설치

### 1. 저장소 클론

```bash
git clone <repository-url>
cd slack-mcp
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

`.env` 파일 생성:

```bash
# Slack Bot Token (xoxb-로 시작하는 토큰)
SLACK_BOT_TOKEN=xoxb-your-bot-token-here

# 선택적: 로그 레벨 설정
LOG_LEVEL=INFO
```

### 4. Cursor MCP 설정

`~/.cursor/mcp.json` 파일에 다음 추가:

```json
{
  "mcpServers": {
    "slack": {
      "command": "python",
      "args": ["/path/to/slack-mcp/slack_mcp_server.py"],
      "env": {
        "SLACK_BOT_TOKEN": "xoxb-your-bot-token-here"
      }
    }
  }
}
```

### 5. 테스트 실행

```bash
python test_slack_mcp.py
```

## 📖 사용법

Cursor에서 새 채팅을 시작하고 다음과 같이 자연어로 Slack 기능을 사용할 수 있습니다:

### 메시지 전송
```
#general 채널에 "안녕하세요!"라는 메시지를 보내주세요
```

### 채널 정보 조회
```
사용 가능한 Slack 채널 목록을 보여주세요
```

### 메시지 히스토리
```
#general 채널의 최근 메시지 10개를 가져와주세요
```

### 다이렉트 메시지
```
사용자 U1234567890에게 "회의 준비 완료"라는 DM을 보내주세요
```

### 메시지 검색
```
"프로젝트"라는 키워드가 포함된 메시지를 검색해주세요
```

## 🛠️ 사용 가능한 도구

| 도구 | 설명 | 파라미터 |
|------|------|----------|
| `send_slack_message` | 채널에 메시지 전송 | `channel`, `text` |
| `get_slack_channels` | 채널 목록 조회 | 없음 |
| `get_slack_channel_history` | 채널 히스토리 조회 | `channel_id`, `limit` |
| `send_slack_direct_message` | 다이렉트 메시지 전송 | `user_id`, `text` |
| `get_slack_users` | 사용자 목록 조회 | 없음 |
| `add_slack_reaction` | 메시지에 반응 추가 | `channel`, `timestamp`, `name` |
| `search_slack_messages` | 메시지 검색 | `query`, `count` |
| `test_slack_connection` | 연결 테스트 | 없음 |

## 📁 프로젝트 구조

```
slack-mcp/
├── 📄 README.md                 # 이 파일
├── 📄 requirements.txt          # Python 패키지 의존성
├── 📄 slack_api.py              # Slack API 클라이언트
├── 📄 slack_mcp_server.py       # FastMCP 서버 구현
├── 📄 test_slack_mcp.py         # 테스트 스크립트
├── 📄 install.py                # 자동 설치 스크립트
├── 📄 mcp_config.json          # MCP 설정 템플릿
├── 📄 env_example.txt          # 환경 변수 템플릿
└── 📄 .gitignore               # Git 무시 파일
```

## 👨‍💻 개발자 가이드

### 개발 환경 설정

```bash
# 개발 모드로 패키지 설치
pip install -r requirements.txt

# 환경 변수 설정
cp env_example.txt .env
# .env 파일을 편집하여 실제 토큰 입력

# 테스트 실행
python test_slack_mcp.py
```

### 새로운 도구 추가

1. `slack_api.py`에 API 메서드 추가
2. `slack_mcp_server.py`에 MCP 도구 데코레이터 추가
3. `test_slack_mcp.py`에 테스트 케이스 추가

### 로깅

서버는 다음 위치에 로그를 기록합니다:
- **콘솔 출력**: 실시간 로그 확인
- **파일 로그**: `slack_mcp.log`

### 디버깅

```bash
# 상세 로그와 함께 테스트 실행
LOG_LEVEL=DEBUG python test_slack_mcp.py

# 서버 직접 실행 (디버깅용)
python slack_mcp_server.py
```

## 🔍 문제 해결

### 자주 발생하는 문제

#### 1. "SLACK_BOT_TOKEN 환경 변수가 설정되지 않았습니다"

**해결책:**
- `.env` 파일에 올바른 토큰이 있는지 확인
- 토큰이 `xoxb-`로 시작하는지 확인

#### 2. "invalid_auth" 에러

**해결책:**
- Slack Bot Token이 유효한지 확인
- 토큰이 올바른 워크스페이스용인지 확인
- 필요한 권한이 모두 부여되었는지 확인

#### 3. "channel_not_found" 에러

**해결책:**
- 채널 ID가 올바른지 확인
- 봇이 해당 채널에 추가되었는지 확인
- 공개 채널인지 또는 봇이 초대되었는지 확인

#### 4. Cursor에서 MCP 도구가 보이지 않음

**해결책:**
- Cursor를 완전히 재시작
- `~/.cursor/mcp.json` 설정 파일 확인
- 파일 경로가 올바른지 확인

### 로그 확인

```bash
# 서버 로그 확인
tail -f slack_mcp.log

# 테스트 로그와 함께 실행
python test_slack_mcp.py 2>&1 | tee debug.log
```

### 수동 연결 테스트

```bash
# Python에서 직접 테스트
python -c "
import asyncio
from slack_api import get_slack_client
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('SLACK_BOT_TOKEN')
client = get_slack_client(token)

async def test():
    result = await client.test_connection()
    print(result)

asyncio.run(test())
"
```

## 📋 과제 요구사항 대비 현황

| 요구사항 | 상태 | 구현 위치 |
|----------|------|-----------|
| **필수 기능** |
| 메시지 전송 | ✅ 완료 | `send_slack_message` |
| 채널 목록 조회 | ✅ 완료 | `get_slack_channels` |
| 채널 히스토리 조회 | ✅ 완료 | `get_slack_channel_history` |
| 다이렉트 메시지 | ✅ 완료 | `send_slack_direct_message` |
| **선택 기능** |
| 사용자 목록 조회 | ✅ 완료 | `get_slack_users` |
| 반응 추가 | ✅ 완료 | `add_slack_reaction` |
| 메시지 검색 | ✅ 완료 | `search_slack_messages` |
| **기술 요구사항** |
| FastMCP 사용 | ✅ 완료 | `slack_mcp_server.py` |
| 에러 처리 | ✅ 완료 | 모든 도구에 try-catch |
| 타입 힌트 | ✅ 완료 | 모든 함수에 타입 힌트 |
| 로깅 | ✅ 완료 | 구조화된 로깅 시스템 |
| 테스트 | ✅ 완료 | `test_slack_mcp.py` |
| 문서화 | ✅ 완료 | 이 README 및 코드 주석 |

## 🏆 추가 구현 사항

과제 요구사항을 넘어서 다음 기능들을 추가로 구현했습니다:

- 🔄 **자동 설치 시스템**: 원클릭 설치 스크립트
- 🧪 **포괄적 테스트**: 모든 기능에 대한 자동화된 테스트
- 📊 **상세한 응답 포맷팅**: 구조화된 성공/실패 응답
- ⚡ **비동기 HTTP 클라이언트**: 고성능 aiohttp 사용
- 🛡️ **세션 관리**: HTTP 세션 재사용으로 성능 최적화
- 🔧 **개발자 도구**: 디버깅 및 문제 해결 도구
- 📁 **완전한 프로젝트 구조**: 생산 준비 완료된 코드 구조

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 🚀 시작하기

```bash
# 1. 프로젝트 클론
git clone <repository-url>
cd slack-mcp

# 2. 자동 설치 실행
python install.py

# 3. Cursor 재시작
# 이제 Cursor에서 Slack MCP 도구를 사용할 수 있습니다!
```

**문제가 있으시면 [Issues](../../issues)에 보고해주세요.** 